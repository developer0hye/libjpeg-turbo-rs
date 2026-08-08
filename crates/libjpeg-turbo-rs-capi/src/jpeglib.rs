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

    /// Mutable access to the backing `Vec` when the source is `Owned`, so the
    /// P4-13 incremental drain can append freshly-pulled body bytes in place.
    fn as_owned_mut(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            JpegSource::Owned(v) => Some(v),
            _ => None,
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

/// Process-global registry mapping the `jvirt_barray_ptr*` array
/// returned by `jpeg_read_coefficients` to the parsed `CoefHandle` it
/// was built alongside. `jpeg_write_coefficients` consults this table
/// so the in-process round-trip path (read coefficients, immediately
/// write coefficients without touching them) can shortcut to the
/// CoefHandle's `JpegCoefficients` and skip rebuilding from individual
/// barray reads. Foreign arrays produced by a stock `transupp` /
/// `jtransform_adjust_parameters` won't appear in the table and fall
/// through to the slower materialise-from-barrays path.
///
/// Pointers are stored as `usize` because raw pointers don't implement
/// `Send`; we cast back to the appropriate pointer type at the call
/// site, which is sound because the lifetimes are managed by the
/// owning cinfo (registry entries are removed in
/// `jpeg_destroy_decompress` / `jpeg_finish_decompress` /
/// `jpeg_abort_decompress` before the storage is freed).
fn coef_array_to_handle_table() -> &'static std::sync::Mutex<std::collections::HashMap<usize, usize>>
{
    static TABLE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, usize>>> =
        std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn coef_register_array(array_ptr: *const c_void, handle_ptr: *const CoefHandle) {
    if let Ok(mut t) = coef_array_to_handle_table().lock() {
        t.insert(array_ptr as usize, handle_ptr as usize);
    }
}

fn coef_lookup_handle(array_ptr: *const c_void) -> Option<*const CoefHandle> {
    coef_array_to_handle_table()
        .lock()
        .ok()
        .and_then(|t| t.get(&(array_ptr as usize)).copied())
        .map(|p| p as *const CoefHandle)
}

fn coef_unregister_array(array_ptr: *const c_void) {
    if let Ok(mut t) = coef_array_to_handle_table().lock() {
        t.remove(&(array_ptr as usize));
    }
}

/// Owned-marker node backing `JpegDecompressPublic::marker_list`. The
/// caller (stock `transupp::jcopy_markers_execute`, etc.) iterates the
/// linked list through the `public` field and reads `data` for the
/// payload — both must stay valid until the cinfo is destroyed. We
/// keep `payload` here so the byte buffer outlives `public.data`.
struct OwnedMarker {
    public: JpegMarkerStructPublic,
    /// Backing storage for `public.data`. Held in a Box so its address
    /// is stable across `marker_list_storage` Vec resizes.
    payload: Box<[u8]>,
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
    /// Held in `Box<CoefHandle>` so the address is pinned across reuses
    /// of the cinfo. `coef_array_to_handle_table` maps the
    /// caller-visible `jvirt_barray_ptr*` array address back to this
    /// CoefHandle so an in-process `jpeg_write_coefficients` can
    /// shortcut to the cached `JpegCoefficients`.
    coefficients: Option<Box<CoefHandle>>,
    /// Caller-visible `jvirt_barray_ptr*` array returned from the most
    /// recent `jpeg_read_coefficients`. Storage is owned by the cinfo's
    /// `JpegMemoryMgr` (JPOOL_IMAGE), so the pointer is valid until the
    /// cinfo is destroyed; we keep it here to deregister from the
    /// global side table on destroy/abort/finish to avoid table leaks.
    coef_array_ptr: *mut c_void,
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
    /// Owned backing for `JpegDecompressPublic::marker_list`. Each
    /// `OwnedMarker` holds both the public C-ABI struct (visible to
    /// callers like stock `transupp::jcopy_markers_execute`) and the
    /// payload bytes the `data` field points into. Populated by
    /// `jpeg_read_header` after parsing the source markers. The list
    /// stays alive until the cinfo is destroyed / `jpeg_abort` is
    /// called (matches stock libjpeg-turbo's marker_list lifetime).
    ///
    /// `Box<OwnedMarker>` (rather than the smaller `OwnedMarker`) is
    /// deliberate: the cinfo's `marker_list` raw pointer points
    /// directly at `OwnedMarker::public`, so each node's address must
    /// stay stable across any future resize / clear / push pattern.
    /// `Box` puts each node in its own pinned heap allocation, which
    /// `Vec<OwnedMarker>` would not.
    #[allow(clippy::vec_box)]
    marker_list_storage: Vec<Box<OwnedMarker>>,
    /// Partial buffer accumulated by `drain_caller_source_mgr` across
    /// suspending retries. Empty when no bridge is active or when the
    /// previous bridge run committed to `JpegSource::Owned`. Held
    /// separately from `source` so a caller's `fill_input_buffer`
    /// returning `FALSE` (suspend) doesn't lose the bytes already
    /// drained via `next_input_byte` advances; the next retry resumes
    /// from the partial accumulator instead of asking the caller for
    /// the same bytes again (which the caller's source manager has
    /// already considered "consumed").
    bridge_partial: Vec<u8>,
    /// `TRUE` once `jpeg_read_header` has completed successfully.
    /// `jpeg_consume_input` consults it so a buffered/progressive
    /// caller polling input after the header is already parsed
    /// doesn't re-trigger header parsing — that would reset
    /// header-derived defaults (`out_color_space`, `comp_info`,
    /// quantize_colors, etc.) and clobber any post-header tweaks the
    /// caller made between `jpeg_read_header` and the next
    /// `consume_input` poll. Cleared in `jpeg_finish_decompress` /
    /// `jpeg_abort_decompress` so a reuse of the same handle re-parses
    /// the new image.
    header_parsed_ok: bool,
    /// Lazily materialised raw-plane cache populated on the first
    /// `jpeg_read_raw_data` call (8-bit baseline/progressive only).
    /// Subsequent calls deliver further iMCU rows from this cache.
    raw_image_cache: Option<libjpeg_turbo_rs::RawImage>,
    /// 12-bit counterpart to `raw_image_cache`. Populated on the first
    /// `jpeg12_read_raw_data` call from
    /// `libjpeg_turbo_rs::raw_data_12::decompress_raw_12`. The 8-bit and
    /// 12-bit caches are mutually exclusive on a single cinfo
    /// (selected by `data_precision`); `raw_rows_consumed` is shared
    /// since its row-count semantics don't depend on element type.
    raw_image_cache_12: Option<libjpeg_turbo_rs::raw_data_12::RawImage12>,
    /// Per-component row cursor into `raw_image_cache` /
    /// `raw_image_cache_12`. Entry `i` is the number of rows already
    /// delivered for component `i`.
    raw_rows_consumed: Vec<usize>,
    /// Bytes from a previously-parsed tables-only abbreviated datastream
    /// (libjpeg.txt §6: SOI + DQT/DHT/DAC + EOI, no SOF/SOS), with the
    /// trailing `FF D9` (EOI) already stripped so a subsequent strip
    /// stream can be spliced after these bytes to form a single
    /// self-contained JPEG. Set by `jpeg_read_header` when it detects a
    /// tables-only input and returns `JPEG_HEADER_TABLES_ONLY`. Consumed
    /// (read; never cleared) by subsequent `jpeg_read_header` calls so
    /// `Decoder::new` sees a complete tables+image stream. The libtiff
    /// `JPEGTables` field flow needs this — libtiff calls
    /// `jpeg_read_header` first on the bare tables blob, then on each
    /// strip's image-without-tables payload.
    tables_only_prefix: Option<Vec<u8>>,
    /// P4-13: `TRUE` when `jpeg_read_header` returned `JPEG_HEADER_OK` from a
    /// *suspending* source manager that had delivered the header (through the
    /// first SOS) but not yet the full body (entropy + remaining scans + EOI).
    /// While set, `jpeg_consume_input` drives the body drain in lock-step with
    /// the source manager (returning `JPEG_REACHED_SOS` / `JPEG_REACHED_EOI` /
    /// `JPEG_SUSPENDED`), and pixel decode is deferred until the body is
    /// complete. Only ever set when `fill_input_buffer` returns `FALSE` mid-body
    /// — so non-suspending consumers (libtiff, Pillow's `mem_src`, djpeg) never
    /// take this path and keep the original fully-buffered behaviour.
    body_incomplete: bool,
    /// Offset into `source` from which `scan_next_boundary` resumes looking for
    /// the next scan / EOI. Starts at the first byte of entropy data (one past
    /// the first SOS header) and advances past each boundary `consume_input`
    /// reports.
    body_scan_cursor: usize,
    /// `TRUE` once the EOI marker has been drained into `source`, i.e. the input
    /// is complete and the buffered decode can run.
    eoi_seen: bool,
}

impl Default for DecompressPrivate {
    fn default() -> Self {
        Self {
            source: JpegSource::None,
            source_mgr: None,
            last_error: CString::new("No error").expect("static"),
            decoded: None,
            coefficients: None,
            coef_array_ptr: std::ptr::null_mut(),
            marker_save: MarkerSaveSettings::default(),
            marker_processors: std::collections::HashMap::new(),
            crop_xoffset: 0,
            crop_width: 0,
            crop_active: false,
            comp_info_storage: Vec::new(),
            marker_list_storage: Vec::new(),
            bridge_partial: Vec::new(),
            header_parsed_ok: false,
            raw_image_cache: None,
            raw_image_cache_12: None,
            raw_rows_consumed: Vec::new(),
            tables_only_prefix: None,
            body_incomplete: false,
            body_scan_cursor: 0,
            eoi_seen: false,
        }
    }
}

impl Drop for DecompressPrivate {
    fn drop(&mut self) {
        // The `jvirt_barray_ptr*` array we returned from
        // `jpeg_read_coefficients` was allocated through this
        // cinfo's `JpegMemoryMgr` (JPOOL_IMAGE) and is freed when the
        // pool is freed. The `CoefHandle` it pointed at lives inside
        // this struct and is about to drop. Remove the global
        // side-table entry now so a future cinfo cannot accidentally
        // get a stale handle pointer if the address happens to be
        // reused by the system allocator.
        if !self.coef_array_ptr.is_null() {
            coef_unregister_array(self.coef_array_ptr as *const c_void);
            self.coef_array_ptr = std::ptr::null_mut();
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

unsafe extern "C" fn default_emit_message(cinfo: *mut c_void, msg_level: c_int) {
    // Mirror libjpeg's `emit_message` contract from jerror.c so callers
    // that follow the documented protocol (counting warnings via
    // `num_warnings`, gating "first warning" printing on
    // `num_warnings == 0`) interoperate correctly:
    //   * msg_level < 0 → warning. Bump `num_warnings`; route to
    //     `output_message` only when this is the first warning OR
    //     trace_level >= 3.
    //   * msg_level >= 0 → trace. Route to `output_message` only when
    //     `msg_level <= trace_level`.
    if cinfo.is_null() {
        return;
    }
    unsafe {
        let err_pp: *const *mut JpegErrorMgr = cinfo as *const *mut JpegErrorMgr;
        let err_ptr: *mut JpegErrorMgr = err_pp.read();
        if err_ptr.is_null() {
            return;
        }
        let err: &mut JpegErrorMgr = &mut *err_ptr;
        if msg_level < 0 {
            // Match libjpeg-turbo's jerror.c::emit_message order: route
            // to `output_message` first (so a custom output hook still
            // sees `num_warnings == 0` on the first warning, the way
            // libjpeg's example callers expect), then increment.
            if err.num_warnings == 0 || err.trace_level >= 3 {
                if let Some(out) = err.output_message {
                    out(cinfo);
                }
            }
            err.num_warnings = err.num_warnings.saturating_add(1);
        } else if msg_level <= err.trace_level {
            if let Some(out) = err.output_message {
                out(cinfo);
            }
        }
    }
}

unsafe extern "C" fn default_output_message(_cinfo: *mut c_void) {
    // No-op by default — real libjpeg routes through stderr.
}

// Upstream `JERR_*` message codes as they resolve at `JPEG_LIB_VERSION = 80`.
//
// The values are positions in `jerror.h`'s `JMESSAGE` list with the
// version-gated entries applied (`JERR_ARITH_NOTIMPL` is v6b-only —
// `#if JPEG_LIB_VERSION < 70`; `JERR_BAD_CROP_SPEC`, `JERR_BAD_DROP_SAMPLING`
// and `JERR_NO_ARITH_TABLE` are v7+). Consumers compare `err->msg_code`
// against the enum from their own `jerror.h`, so these must stay exact.
// Coverage is uneven: `tests/capi_classic_dest_ownership.rs` cross-checks
// `JERR_BUFFER_SIZE` and `JERR_FILE_WRITE` against a reference v8 build,
// `tests/capi_classic_lifecycle_pathological.rs` pins `JERR_CANT_SUSPEND`
// against upstream's own enum, and `JERR_OUT_OF_MEMORY` is unpinned — no test
// can force the `malloc` failure that reaches it.
/// "Buffer passed to JPEG library is too small"
const JERR_BUFFER_SIZE: c_int = 24;
/// "Suspension not allowed here"
const JERR_CANT_SUSPEND: c_int = 25;
/// "Output file write error --- out of disk space?"
const JERR_FILE_WRITE: c_int = 38;
/// "Insufficient memory (case %d)"
const JERR_OUT_OF_MEMORY: c_int = 56;

/// Invoke `cinfo->err->error_exit(cinfo)` with the given `msg_code`,
/// mirroring upstream's `ERREXIT` macro family in `jerror.h`.
///
/// libjpeg's contract (libjpeg.txt §3) is that whenever the library
/// detects an unrecoverable error during a public-API call (corrupt
/// stream, bogus marker length, out-of-range parameter, …), it must
/// route through `cinfo->err->error_exit(cinfo)`. Consumers override
/// `error_exit` with a `setjmp`/`longjmp` handler so the call returns
/// control to user code rather than aborting the process.
///
/// Most consumer overrides longjmp out and never return; this helper
/// still returns cleanly if a custom handler does return (which
/// violates the libjpeg contract, but defensive code is cheap), so the
/// caller can fall through to its own error-return path.
fn invoke_error_exit(cinfo: *mut c_void, msg_code: c_int) {
    if cinfo.is_null() {
        return;
    }
    // SAFETY: caller guarantees `cinfo` is a valid `j_common_ptr`-shaped
    // struct whose first pointer-sized field is the `err` slot.
    unsafe {
        let err_pp: *const *mut JpegErrorMgr = cinfo as *const *mut JpegErrorMgr;
        let err_ptr: *mut JpegErrorMgr = err_pp.read();
        if err_ptr.is_null() {
            return;
        }
        let err: &mut JpegErrorMgr = &mut *err_ptr;
        err.msg_code = msg_code;
        if let Some(exit) = err.error_exit {
            exit(cinfo);
        }
    }
}

/// `invoke_error_exit` for the upstream `ERREXIT1` family: also publishes the
/// single integer parameter the message's `%d` substitutes.
///
/// Without it `format_message` renders whatever integer happened to be left in
/// `msg_parm`, which is exactly the kind of divergence a drop-in replacement
/// must not have.
fn invoke_error_exit_parm(cinfo: *mut c_void, msg_code: c_int, parm0: c_int) {
    if cinfo.is_null() {
        return;
    }
    // SAFETY: caller guarantees `cinfo` is a valid `j_common_ptr`-shaped
    // struct whose first pointer-sized field is the `err` slot.
    unsafe {
        let err_pp: *const *mut JpegErrorMgr = cinfo as *const *mut JpegErrorMgr;
        let err_ptr: *mut JpegErrorMgr = err_pp.read();
        if !err_ptr.is_null() {
            let err: &mut JpegErrorMgr = &mut *err_ptr;
            // `msg_parm` is the C union's raw bytes; `i[0]` is its first int.
            let bytes: [u8; std::mem::size_of::<c_int>()] = parm0.to_ne_bytes();
            err.msg_parm[..bytes.len()].copy_from_slice(&bytes);
        }
    }
    invoke_error_exit(cinfo, msg_code);
}

unsafe extern "C" fn default_format_message(cinfo: *mut c_void, buffer: *mut u8) {
    if buffer.is_null() {
        return;
    }
    // Mirror libjpeg-turbo's jerror.c::format_message: look up the
    // message text by `msg_code` in the per-cinfo `jpeg_message_table`
    // or `addon_message_table`, then substitute parameters from
    // `msg_parm` using printf-style format specifiers.
    //
    // Upstream contract (jerror.c:178-196): if the first `%X` in the
    // message is `%s`, the *only* parameter is `err->msg_parm.s`
    // (the string union arm). Otherwise, parameters come from
    // `err->msg_parm.i[0..7]` (eight ints). Mixing the two in one
    // message is not supported by upstream and we follow suit.
    let mut msgtext: *const u8 = std::ptr::null();
    let mut msg_parm_bytes: [u8; JMSG_STR_PARM_MAX] = [0u8; JMSG_STR_PARM_MAX];
    let mut have_err: bool = false;
    if !cinfo.is_null() {
        unsafe {
            let err_pp: *const *mut JpegErrorMgr = cinfo as *const *mut JpegErrorMgr;
            let err_ptr: *mut JpegErrorMgr = err_pp.read();
            if !err_ptr.is_null() {
                let err: &JpegErrorMgr = &*err_ptr;
                have_err = true;
                msg_parm_bytes = err.msg_parm;
                let code: c_int = err.msg_code;
                if code > 0 && !err.jpeg_message_table.is_null() && code <= err.last_jpeg_message {
                    msgtext = err.jpeg_message_table.add(code as usize).read();
                } else if !err.addon_message_table.is_null()
                    && code >= err.first_addon_message
                    && code <= err.last_addon_message
                {
                    let idx: c_int = code - err.first_addon_message;
                    msgtext = err.addon_message_table.add(idx as usize).read();
                }
            }
        }
    }

    // Resolve msgtext to a byte slice WITHOUT the trailing NUL.
    let format_bytes: &[u8] = if !msgtext.is_null() {
        // SAFETY: msgtext came from one of the message tables, both of
        // which contain `'static` NUL-terminated byte strings.
        unsafe {
            let mut len: usize = 0;
            while *msgtext.add(len) != 0 {
                len += 1;
            }
            std::slice::from_raw_parts(msgtext, len)
        }
    } else {
        b"libjpeg-turbo-rs: bogus message code"
    };

    // Reinterpret the msg_parm bytes as the C union view.
    // Note: the `i` arm is `c_int[8]` = 32 bytes (LP64) / 32 bytes (LLP64).
    // Reading as native-endian via byte copy is correct for both alignments
    // because we made a stack copy (msg_parm_bytes) to a 4-byte-aligned
    // location; if a future caller proves otherwise, switch to byte-wise
    // assembly.
    let int_args: [c_int; 8] = if have_err {
        let mut ints: [c_int; 8] = [0; 8];
        for (i, slot) in ints.iter_mut().enumerate() {
            let off = i * std::mem::size_of::<c_int>();
            if off + std::mem::size_of::<c_int>() <= JMSG_STR_PARM_MAX {
                let mut b: [u8; 4] = [0; 4];
                b.copy_from_slice(&msg_parm_bytes[off..off + 4]);
                *slot = c_int::from_ne_bytes(b);
            }
        }
        ints
    } else {
        [0; 8]
    };
    let string_arg: &[u8] = {
        // The `s` arm is a NUL-terminated char[JMSG_STR_PARM_MAX].
        let nul: usize = msg_parm_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(JMSG_STR_PARM_MAX);
        // SAFETY-equivalent: stack copy is local, slice is in-bounds.
        // We have to allocate to escape the stack-local — caller of this
        // helper consumes a slice with no lifetime tied to msg_parm_bytes.
        // We'll route through a shared buffer via the caller's stack.
        unsafe { std::slice::from_raw_parts(msg_parm_bytes.as_ptr(), nul) }
    };

    // Decide string-mode vs int-mode by scanning for the first `%X` in
    // the message, matching jerror.c:181-186.
    let mut is_string: bool = false;
    let mut i: usize = 0;
    while i < format_bytes.len() {
        if format_bytes[i] == b'%' && i + 1 < format_bytes.len() {
            if format_bytes[i + 1] == b's' {
                is_string = true;
            }
            break;
        }
        i += 1;
    }

    // Format into a stack buffer sized to JMSG_LENGTH_MAX (matches the
    // C contract: caller passes a buffer of at least JMSG_LENGTH_MAX
    // bytes, and snprintf truncates to fit).
    let mut out: [u8; JMSG_LENGTH_MAX] = [0u8; JMSG_LENGTH_MAX];
    let written: usize = if is_string {
        // String-mode: msg_parm.s is the only argument. Bind a single-
        // element slice so the parser walks args by index just like
        // int-mode.
        snprintf_jpeg(&mut out, format_bytes, Some(string_arg), &[])
    } else {
        snprintf_jpeg(&mut out, format_bytes, None, &int_args)
    };

    // Copy to caller's buffer (caller-allocated, must be ≥ JMSG_LENGTH_MAX).
    // Always NUL-terminate.
    let copy_len: usize = (written + 1).min(JMSG_LENGTH_MAX);
    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr(), buffer, copy_len);
        // Belt-and-braces NUL terminator at the end.
        if copy_len > 0 {
            *buffer.add(copy_len - 1) = 0;
        }
    }
}

/// Minimal printf-style formatter covering the specifiers libjpeg-turbo's
/// jerror.h actually uses (`%s %d %u %x %X %c %02d %3d %4u %02x %04x %%`).
///
/// Returns the number of bytes written to `out` (not including any
/// trailing NUL). The output is truncated at `out.len() - 1` to leave
/// room for a NUL the caller will write.
///
/// `string_arg` is the single string parameter when format contains
/// `%s`; `int_args` are consumed positionally for non-string specifiers.
/// Mixing `%s` with integer specifiers is not supported (matches the
/// jerror.c contract).
fn snprintf_jpeg(
    out: &mut [u8],
    format: &[u8],
    string_arg: Option<&[u8]>,
    int_args: &[c_int],
) -> usize {
    let cap_excl_nul: usize = out.len().saturating_sub(1);
    let mut written: usize = 0;
    let mut int_idx: usize = 0;
    let mut i: usize = 0;

    let push = |out: &mut [u8], written: &mut usize, b: u8| {
        if *written < cap_excl_nul {
            out[*written] = b;
            *written += 1;
        }
    };
    let push_bytes = |out: &mut [u8], written: &mut usize, bytes: &[u8]| {
        for &b in bytes {
            push(out, written, b);
        }
    };

    while i < format.len() {
        let b: u8 = format[i];
        if b != b'%' {
            push(out, &mut written, b);
            i += 1;
            continue;
        }
        i += 1;
        if i >= format.len() {
            // Trailing `%` — emit literally.
            push(out, &mut written, b'%');
            break;
        }
        if format[i] == b'%' {
            push(out, &mut written, b'%');
            i += 1;
            continue;
        }
        // Optional flag chars. We honour `0` (zero-pad); silently
        // accept-and-ignore the rest because jerror.h doesn't use them.
        let mut flag_zero: bool = false;
        while i < format.len() {
            match format[i] {
                b'0' => {
                    flag_zero = true;
                    i += 1;
                }
                b'-' | b'+' | b' ' | b'#' => {
                    i += 1;
                }
                _ => break,
            }
        }
        // Optional width.
        let mut width: usize = 0;
        while i < format.len() && format[i].is_ascii_digit() {
            width = width * 10 + (format[i] - b'0') as usize;
            i += 1;
        }
        if i >= format.len() {
            break;
        }
        let spec: u8 = format[i];
        i += 1;
        let formatted: Vec<u8> = match spec {
            b's' => string_arg.unwrap_or(&[]).to_vec(),
            b'd' | b'i' => {
                let v: c_int = int_args.get(int_idx).copied().unwrap_or(0);
                int_idx += 1;
                v.to_string().into_bytes()
            }
            b'u' => {
                let v: c_int = int_args.get(int_idx).copied().unwrap_or(0);
                int_idx += 1;
                (v as c_uint).to_string().into_bytes()
            }
            b'x' => {
                let v: c_int = int_args.get(int_idx).copied().unwrap_or(0);
                int_idx += 1;
                format!("{:x}", v as c_uint).into_bytes()
            }
            b'X' => {
                let v: c_int = int_args.get(int_idx).copied().unwrap_or(0);
                int_idx += 1;
                format!("{:X}", v as c_uint).into_bytes()
            }
            b'c' => {
                let v: c_int = int_args.get(int_idx).copied().unwrap_or(0);
                int_idx += 1;
                vec![(v & 0xFF) as u8]
            }
            other => {
                // Unrecognised specifier — emit raw `%X` so the caller
                // can spot the mismatch.
                push(out, &mut written, b'%');
                push(out, &mut written, other);
                continue;
            }
        };
        // Apply width / zero-padding (right-justified only — `-` flag
        // is parsed but ignored; jerror.h never uses it).
        if formatted.len() < width {
            let pad: u8 = if flag_zero { b'0' } else { b' ' };
            for _ in 0..(width - formatted.len()) {
                push(out, &mut written, pad);
            }
        }
        push_bytes(out, &mut written, &formatted);
    }
    written
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
    crate::unwind_guard!(std::ptr::null_mut(), {
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
    })
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
    crate::unwind_guard!((), {
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
    })
}

/// `jpeg_destroy_decompress(cinfo)` — free the Rust-side private state.
#[no_mangle]
pub extern "C" fn jpeg_destroy_decompress(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
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
    })
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

/// Drain a caller-installed source manager (`cinfo.src`) into a
/// contiguous owned buffer.
///
/// Pillow's `_imaging.so` (and similarly libtiff's libjpeg consumer)
/// installs its own `jpeg_source_mgr` directly without going through
/// our `jpeg_mem_src`/`jpeg_stdio_src`. Without bridging that
/// installation, our decoder sees `JpegSource::None` and can't make
/// progress.
///
/// Protocol per `references/libjpeg-turbo/src/jdatasrc.c`:
///   1. `init_source(cinfo)` if non-NULL — gives the caller a chance
///      to pre-load `next_input_byte` / `bytes_in_buffer`.
///   2. Drain the current public buffer (PIL's `jpeg_buffer_src` already
///      pre-loads it before our `jpeg_read_header` runs, so this step
///      typically captures the whole image).
///   3. If we haven't seen `FF D9` (EOI), call `fill_input_buffer` and
///      drain again — repeat until EOI or `fill_input_buffer` returns
///      FALSE (suspension) or the safety cap (256 MiB) is reached.
///
/// Returns `None` when no source manager is present, when `init_source`
/// is missing AND `bytes_in_buffer` is 0, or when the safety cap is
/// hit. Callers should fall through to the existing
/// `JpegSource::None` "no JPEG source attached" error path in that
/// case.
///
/// # Safety
/// `c` must be a live `JpegDecompressPublic` that the caller continues
/// to own across this call. `c.src` (if non-NULL) must point to a
/// `jpeg_source_mgr` that follows the libjpeg ABI, including ABI-
/// compatible callback pointers.
unsafe fn drain_caller_source_mgr(
    c: &mut JpegDecompressPublic,
    priv_state: &mut DecompressPrivate,
) -> Option<Vec<u8>> {
    let src_ptr: *mut JpegSourceMgr = c.src;
    if src_ptr.is_null() {
        return None;
    }
    let cinfo_ptr: *mut c_void = c as *mut JpegDecompressPublic as *mut c_void;
    // SAFETY: caller asserts `c.src` points at a live JpegSourceMgr.
    let src: &mut JpegSourceMgr = unsafe { &mut *src_ptr };

    // Step 1: init_source. Only invoke once per bridge cycle —
    // otherwise a non-blocking caller's init_source might reset its
    // internal stream cursor on every retry. We detect "first call"
    // by `bridge_partial.is_empty()` (no prior partial drain in flight).
    let is_first_call: bool = priv_state.bridge_partial.is_empty();
    if is_first_call {
        if let Some(init) = src.init_source {
            // SAFETY: caller-provided callback honoring the libjpeg
            // `void (*init_source)(j_decompress_ptr)` prototype.
            unsafe {
                init(cinfo_ptr);
            }
        }
    }

    // Resume from a prior suspended drain if there was one.
    let mut accumulator: Vec<u8> = std::mem::take(&mut priv_state.bridge_partial);
    // Cap accumulation at 256 MiB. A misbehaving callback that keeps
    // returning TRUE without setting `bytes_in_buffer = 0` would
    // otherwise loop forever; the cap forces a clean abort instead.
    const MAX_BYTES: usize = 256 * 1024 * 1024;

    loop {
        // Step 2a: drain whatever is currently in the public buffer.
        if src.bytes_in_buffer > 0 && !src.next_input_byte.is_null() {
            // SAFETY: caller's ABI promises `next_input_byte` is valid
            // for `bytes_in_buffer` reads.
            let chunk: &[u8] =
                unsafe { std::slice::from_raw_parts(src.next_input_byte, src.bytes_in_buffer) };
            accumulator.extend_from_slice(chunk);
            // SAFETY: advancing within the asserted-valid window.
            src.next_input_byte = unsafe { src.next_input_byte.add(src.bytes_in_buffer) };
            src.bytes_in_buffer = 0;
        }

        // Step 2b: stop if we've seen the End-Of-Image marker (FF D9).
        // libjpeg's contract guarantees a JPEG stream always ends with
        // EOI; once we've captured it, the decoder doesn't need more
        // bytes regardless of what the source manager has buffered.
        let saw_eoi: bool = accumulator.len() >= 2
            && accumulator[accumulator.len() - 2] == 0xFF
            && accumulator[accumulator.len() - 1] == 0xD9;
        if saw_eoi {
            return Some(accumulator);
        }

        if accumulator.len() > MAX_BYTES {
            return None;
        }

        // Step 3: fetch more bytes from the caller.
        let fill: unsafe extern "C" fn(*mut c_void) -> CBoolean = match src.fill_input_buffer {
            Some(f) => f,
            // No callback and no EOI yet → end of stream from a
            // pre-loaded buffer (typical static-source case).
            None => return Some(accumulator),
        };
        // SAFETY: caller-supplied callback. A FALSE return means
        // "suspended; come back later". Park the partial accumulator
        // in `priv_state.bridge_partial` so the next retry resumes
        // from the same prefix instead of dropping the bytes whose
        // `next_input_byte` advance the caller's source manager has
        // already counted as consumed.
        let ok: CBoolean = unsafe { fill(cinfo_ptr) };
        if ok == 0 {
            priv_state.bridge_partial = accumulator;
            return None;
        }
        if src.bytes_in_buffer == 0 {
            // Successful fill but no bytes — treat as end of stream.
            return Some(accumulator);
        }
    }
}

/// Outcome of a single incremental pull from the caller's live source manager.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum PullResult {
    /// New bytes were appended to the accumulator.
    Progressed,
    /// `fill_input_buffer` returned `FALSE` — the source is dry right now
    /// (`JPEG_SUSPENDED`); the caller should retry after delivering more.
    Suspended,
    /// No callback, or a successful fill produced zero bytes — end of stream.
    Ended,
}

/// Pull one chunk from the caller's live source manager into `accumulator`:
/// drain whatever is currently in the public buffer, or — if it is empty —
/// call `fill_input_buffer` once. Drives a suspending source in lock-step for
/// `jpeg_consume_input` (P4-13), mirroring how `drain_caller_source_mgr`
/// consumes the public buffer but a single step at a time so suspension can be
/// surfaced between scan boundaries.
unsafe fn pull_more_from_source_mgr(
    c: &mut JpegDecompressPublic,
    accumulator: &mut Vec<u8>,
) -> PullResult {
    let src_ptr: *mut JpegSourceMgr = c.src;
    if src_ptr.is_null() {
        return PullResult::Ended;
    }
    // SAFETY: caller asserts `c.src` points at a live JpegSourceMgr.
    let src: &mut JpegSourceMgr = unsafe { &mut *src_ptr };
    let cinfo_ptr: *mut c_void = c as *mut JpegDecompressPublic as *mut c_void;

    // Drain whatever is already buffered first.
    if src.bytes_in_buffer > 0 && !src.next_input_byte.is_null() {
        // SAFETY: the ABI promises `next_input_byte` is valid for
        // `bytes_in_buffer` reads.
        let chunk: &[u8] =
            unsafe { std::slice::from_raw_parts(src.next_input_byte, src.bytes_in_buffer) };
        accumulator.extend_from_slice(chunk);
        src.next_input_byte = unsafe { src.next_input_byte.add(src.bytes_in_buffer) };
        src.bytes_in_buffer = 0;
        return PullResult::Progressed;
    }

    // Buffer empty: ask the caller for more.
    let fill: unsafe extern "C" fn(*mut c_void) -> CBoolean = match src.fill_input_buffer {
        Some(f) => f,
        None => return PullResult::Ended,
    };
    let ok: CBoolean = unsafe { fill(cinfo_ptr) };
    if ok == 0 {
        return PullResult::Suspended;
    }
    if src.bytes_in_buffer == 0 || src.next_input_byte.is_null() {
        return PullResult::Ended;
    }
    // SAFETY: as above.
    let chunk: &[u8] =
        unsafe { std::slice::from_raw_parts(src.next_input_byte, src.bytes_in_buffer) };
    accumulator.extend_from_slice(chunk);
    src.next_input_byte = unsafe { src.next_input_byte.add(src.bytes_in_buffer) };
    src.bytes_in_buffer = 0;
    PullResult::Progressed
}

/// Finish draining a `body_incomplete` (P4-13) stream to EOI, resuming the live
/// source manager and counting each later SOS into `input_scan_number`. Returns
/// `true` once the body is complete (`eoi_seen` set, `body_incomplete` cleared),
/// or `false` if the source suspended (caller should report suspension and be
/// retried) or the 256 MiB cap tripped. Shared by `jpeg_start_decompress`
/// (non-buffered), `jpeg_read_coefficients`, and `jpeg_read_raw_data`, which all
/// need the complete buffer before parsing.
fn finish_body_drain(c: &mut JpegDecompressPublic, priv_state: &mut DecompressPrivate) -> bool {
    if !priv_state.body_incomplete {
        return true;
    }
    loop {
        let cursor: usize = priv_state.body_scan_cursor;
        match priv_state
            .source
            .as_bytes()
            .map(|b| scan_next_boundary(b, cursor))
        {
            Some(MarkerBoundary::Eoi(off)) => {
                priv_state.body_scan_cursor = off;
                priv_state.eoi_seen = true;
                priv_state.body_incomplete = false;
                rebuild_marker_list_from_source(c, priv_state);
                return true;
            }
            Some(MarkerBoundary::Sos(off)) => {
                priv_state.body_scan_cursor = off;
                c.input_scan_number = c.input_scan_number.saturating_add(1);
                continue;
            }
            // NeedMore: resume the next scan from where this one stalled (avoids
            // O(n²) rescans on a small-packet source). `None` (no source bytes)
            // leaves the cursor unchanged.
            Some(MarkerBoundary::NeedMore(resume)) => priv_state.body_scan_cursor = resume,
            None => {}
        }
        let pull: PullResult = match priv_state.source.as_owned_mut() {
            // SAFETY: `c` and `priv_state` are disjoint allocations.
            Some(buf) => unsafe { pull_more_from_source_mgr(c, buf) },
            None => return false,
        };
        match pull {
            PullResult::Progressed => {
                if priv_state.source.as_bytes().map_or(0, <[u8]>::len) > P4_13_MAX_BODY_BYTES {
                    // Cap tripped: clear `body_incomplete` so a retry can't
                    // re-enter the drain and append past the cap (bounds memory).
                    priv_state.body_incomplete = false;
                    priv_state.last_error =
                        CString::new("jpeg decode: suspended body exceeds 256 MiB cap")
                            .unwrap_or_default();
                    return false;
                }
            }
            PullResult::Suspended => return false,
            PullResult::Ended => {
                priv_state.eoi_seen = true;
                priv_state.body_incomplete = false;
                rebuild_marker_list_from_source(c, priv_state);
                return true;
            }
        }
    }
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

/// If `bytes` is a tables-only abbreviated JPEG datastream per
/// libjpeg.txt §6 (SOI, then zero or more DQT/DHT/DAC/COM/APPn/DRI
/// markers, then EOI, with no SOF or SOS), return `Some(prefix)` where
/// `prefix` is the byte run from the SOI through the last marker
/// before EOI — i.e. the input with the trailing `FF D9` (EOI)
/// stripped. The caller can splice this prefix in front of a strip
/// stream that starts with `FF D8` (after dropping its own SOI) to
/// form a single self-contained JPEG that `Decoder::new` parses
/// normally.
///
/// Returns `None` if the input isn't a clean tables-only datastream
/// (e.g. it has an SOF marker, or it's truncated, or it has a malformed
/// marker length). The caller falls back to the existing decode path.
///
/// JPEG marker bytes recognised here:
///   * `FF D8` — SOI (start)
///   * `FF D9` — EOI (end)
///   * `FF DA` — SOS (rejects: not tables-only)
///   * `FF C0..FF CF` minus `{C4, C8, CC}` — SOF variants (rejects)
///   * `FF C4` (DHT), `FF CC` (DAC), `FF DB` (DQT), `FF DD` (DRI),
///     `FF FE` (COM), `FF E0..FF EF` (APPn) — all valid in tables-only
///   * `FF D0..FF D7` (RST0..RST7), `FF 01` (TEM) — no length payload,
///     valid in tables-only (though unusual)
fn detect_tables_only(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() < 4 {
        return None;
    }
    if bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }

    let mut i: usize = 2;
    let mut saw_eoi: bool = false;
    while i + 1 < bytes.len() {
        // Skip any FF padding the encoder may have inserted between
        // markers (some libjpeg-turbo encoders write FF FF as filler).
        while i < bytes.len() && bytes[i] == 0xFF && bytes.get(i + 1) == Some(&0xFF) {
            i += 1;
        }
        if i + 1 >= bytes.len() || bytes[i] != 0xFF {
            return None;
        }
        let marker: u8 = bytes[i + 1];
        i += 2;

        match marker {
            0xD9 => {
                saw_eoi = true;
                break;
            }
            0xD8 => return None, // unexpected duplicate SOI
            0xDA => return None, // SOS: not tables-only
            // SOF range: C0..CF minus C4 (DHT), C8 (reserved JPG), CC (DAC).
            0xC0..=0xCF if marker != 0xC4 && marker != 0xC8 && marker != 0xCC => {
                return None;
            }
            // No-payload markers (RST0..RST7, TEM).
            0xD0..=0xD7 | 0x01 => {}
            // Marker with 2-byte big-endian length (covers DQT/DHT/DAC,
            // DRI, COM, APPn, and any other length-bearing marker).
            _ => {
                if i + 2 > bytes.len() {
                    return None;
                }
                let len: usize = ((bytes[i] as usize) << 8) | (bytes[i + 1] as usize);
                if len < 2 || i + len > bytes.len() {
                    return None;
                }
                i += len;
            }
        }
    }

    if !saw_eoi {
        return None;
    }
    // i now points just past the EOI bytes (FF D9). Strip them so the
    // returned prefix can be spliced in front of a strip stream's body
    // (after the strip's leading FF D8) to form a complete JPEG.
    Some(bytes[..i - 2].to_vec())
}

/// Peek at the JPEG header to populate `cinfo.image_width` etc. without
/// triggering a full decode. Uses the Rust-side `Decoder::new` which
/// only parses markers up to (but not including) entropy-coded data.
///
/// `require_image` mirrors libjpeg's `jpeg_read_header(cinfo, require_image)`
/// flag (libjpeg.txt §6, jdapimin.c::jpeg_read_header). When the input is
/// a tables-only abbreviated datastream:
///   * `require_image == TRUE`  — invoke `error_exit` with `JERR_NO_IMAGE`
///     so a `setjmp`/`longjmp` consumer recovers cleanly. The classic
///     contract for callers that don't care about tables: "I asked for
///     a real image header; you didn't give me one; fail loudly."
///   * `require_image == FALSE` — return `JPEG_HEADER_TABLES_ONLY` so
///     the caller can read the tables and use them on a subsequent
///     image-only datastream (the libtiff `JPEGTables` flow).
#[no_mangle]
pub extern "C" fn jpeg_read_header(cinfo: *mut c_void, require_image: CBoolean) -> c_int {
    crate::unwind_guard!(JPEG_SUSPENDED, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return JPEG_SUSPENDED,
        };
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return JPEG_SUSPENDED,
        };

        // If `jpeg_mem_src` / `jpeg_stdio_src` were called, `priv_state.source`
        // is already populated. Otherwise — typical of Pillow's `_imaging.so`
        // and libtiff's libjpeg consumer, both of which install their own
        // `jpeg_source_mgr` directly — drain the caller-installed source
        // through its `fill_input_buffer` callback into an `Owned` buffer
        // so the rest of the decode path proceeds normally. The bridge
        // resumes across suspending retries via
        // `priv_state.bridge_partial`, so a non-blocking source feeding
        // bytes one chunk at a time eventually succeeds without losing
        // already-drained input.
        if priv_state.source.as_bytes().is_none() {
            // SAFETY: c is a live JpegDecompressPublic; if c.src is non-NULL,
            // the caller asserts it points at a libjpeg-ABI source manager.
            match unsafe { drain_caller_source_mgr(c, priv_state) } {
                Some(drained) => {
                    priv_state.source = JpegSource::Owned(drained);
                }
                None => {
                    // The source manager suspended (`fill_input_buffer` → FALSE)
                    // before EOI. P4-13: if the bytes drained so far already
                    // contain a complete header (through the first SOS), promote
                    // to `JPEG_HEADER_OK` now and let `jpeg_consume_input` drive
                    // the remaining body in lock-step with the suspending source.
                    // Otherwise leave the partial parked in `bridge_partial` and
                    // fall through to `JPEG_SUSPENDED` (header not yet complete).
                    // Non-suspending sources never return `None` here, so libtiff
                    // / djpeg / `mem_src` keep the original fully-buffered path.
                    if let Some(sos_end) = find_first_sos(&priv_state.bridge_partial) {
                        let prefix: Vec<u8> = std::mem::take(&mut priv_state.bridge_partial);
                        priv_state.source = JpegSource::Owned(prefix);
                        priv_state.body_incomplete = true;
                        priv_state.body_scan_cursor = sos_end;
                        priv_state.eoi_seen = false;
                        // Upstream sets input_scan_number to 1 once the first SOS
                        // is reached; mirror that so a buffered-image consumer
                        // reads a correct scan index (jpeg_consume_input bumps it
                        // per subsequent SOS).
                        c.input_scan_number = 1;
                    }
                }
            }
        }

        let raw_bytes: &[u8] = match priv_state.source.as_bytes() {
            Some(b) if b.len() >= 2 => b,
            _ => {
                priv_state.last_error =
                    CString::new("jpeg_read_header: no JPEG source attached").unwrap_or_default();
                return JPEG_SUSPENDED;
            }
        };

        // Tables-only abbreviated datastream (libjpeg.txt §6) detection.
        // libtiff's COMPRESSION_JPEG path calls `jpeg_read_header` on its
        // `JPEGTables` blob first (SOI + DQT/DHT + EOI, no SOF/SOS), expects
        // `JPEG_HEADER_TABLES_ONLY` (= 2), and *then* re-points the source
        // manager at strip data and calls `jpeg_read_header` again to parse
        // the actual image header — which is missing tables that the strip
        // data inherits from the cached `JPEGTables`. The shim emulates this
        // contract by stashing the tables-only bytes (minus the trailing EOI)
        // in `priv_state.tables_only_prefix` and splicing them in front of
        // the next strip's body so `Decoder::new` sees a single self-contained
        // JPEG.
        //
        // Detection runs on every fresh drain (not gated on
        // `tables_only_prefix.is_none()`) so a cinfo reused across multiple
        // images can refresh the cached tables on each new tables-only
        // datastream — stock libjpeg's `jpeg_read_header` accepts repeated
        // tables-only reads and updates the cached tables in place. After
        // detection we reset `priv_state.source` to `None` so the next
        // `jpeg_read_header` call re-drains the caller's source manager and
        // picks up the strip bytes libtiff has now installed.
        if let Some(prefix) = detect_tables_only(raw_bytes) {
            // libjpeg.txt §6 / jdapimin.c::jpeg_read_header: when the
            // caller passed `require_image = TRUE`, a tables-only blob is
            // *not* a valid response — invoke `error_exit` with
            // `JERR_NO_IMAGE` (upstream code 53 at `JPEG_LIB_VERSION=80`,
            // verified empirically against the submodule's `jerror.h`).
            // A `setjmp`/`longjmp` consumer recovers; a default handler
            // aborts. The cached prefix is *not* updated in this branch
            // because the consumer asked for an image and we're delivering
            // the documented error.
            if require_image != 0 {
                priv_state.last_error = CString::new(
                    "jpeg_read_header: JERR_NO_IMAGE — caller passed require_image=TRUE \
                 but the input is a tables-only abbreviated datastream",
                )
                .unwrap_or_default();
                invoke_error_exit(cinfo, 53);
                // Defensive return if a non-conforming handler returns
                // without longjmp-ing (violates the libjpeg contract).
                return JPEG_SUSPENDED;
            }
            priv_state.tables_only_prefix = Some(prefix);
            priv_state.source = JpegSource::None;
            priv_state.last_error = CString::new("No error").unwrap_or_default();
            return JPEG_HEADER_TABLES_ONLY;
        }

        // If we have a stashed tables-only prefix and the new input starts
        // with SOI, splice them so `Decoder::new` sees the tables + image
        // as one stream. The splice is unconditional on prefix presence —
        // libjpeg's table-replacement contract is "the most recent table
        // in source order wins for each `(class, destination)` slot", so
        // an image carrying its own DQT/DHT (a "partial abbreviated"
        // image, or even a fully self-contained JPEG run on a cinfo with
        // a stale prefix) simply overrides whatever the cached prefix
        // installed — no incorrect-decode risk. We persist the spliced
        // bytes back into `priv_state.source` so the rest of the decode
        // path (`jpeg_start_decompress`, `jpeg_read_raw_data`, …) reads
        // from the unified stream.
        let needs_splice: bool = priv_state.tables_only_prefix.is_some()
            && raw_bytes.len() >= 2
            && raw_bytes[0] == 0xFF
            && raw_bytes[1] == 0xD8;
        if needs_splice {
            // `raw_bytes` borrows from `priv_state.source.as_bytes()`. We're
            // about to replace that backing buffer, so build the combined
            // Vec first via owned copies; then assign.
            let prefix_clone: Vec<u8> = priv_state.tables_only_prefix.as_deref().unwrap().to_vec();
            let strip_body: Vec<u8> = raw_bytes[2..].to_vec();
            let mut combined: Vec<u8> = Vec::with_capacity(prefix_clone.len() + strip_body.len());
            combined.extend_from_slice(&prefix_clone);
            combined.extend_from_slice(&strip_body);
            priv_state.source = JpegSource::Owned(combined);
        }
        // P4-13: re-derive the body scan cursor from the FINAL source buffer.
        // If a tables-only prefix was spliced in just above, the first-SOS
        // offset shifted by `prefix.len() - 2`; recomputing here keeps the
        // incremental drain resuming at the real entropy start regardless of
        // splicing. A no-op for the common (no-splice) case — same offset.
        if priv_state.body_incomplete {
            if let Some(sos_end) = priv_state.source.as_bytes().and_then(find_first_sos) {
                priv_state.body_scan_cursor = sos_end;
            }
        }
        let src_bytes: &[u8] = match priv_state.source.as_bytes() {
            Some(b) if b.len() >= 2 => b,
            _ => {
                priv_state.last_error =
                    CString::new("jpeg_read_header: no JPEG source after splice")
                        .unwrap_or_default();
                return JPEG_SUSPENDED;
            }
        };
        // P4-13: when the body is still being drained from a suspending source,
        // the buffered prefix ends just after the first SOS header (no entropy /
        // EOI). Append a synthetic EOI for header parsing only, so `read_markers`
        // terminates cleanly after recording the frame header instead of erroring
        // on truncation — a progressive stream otherwise scans past SOS-1 hunting
        // for more scans and hits end-of-buffer. The actual pixel decode later
        // runs on the complete (real-EOI) buffer, never this synthetic one. All
        // public header fields come from markers before SOS-1, so they are exact.
        let ends_with_eoi: bool = src_bytes.len() >= 2
            && src_bytes[src_bytes.len() - 2] == 0xFF
            && src_bytes[src_bytes.len() - 1] == 0xD9;
        let needs_synthetic_eoi: bool = priv_state.body_incomplete && !ends_with_eoi;
        let parse_holder: Vec<u8> = if needs_synthetic_eoi {
            // Parse exactly the through-SOS prefix (everything up to the first
            // byte of scan-1 entropy, `body_scan_cursor`) plus the synthetic
            // EOI — never the partial entropy bytes that may have arrived before
            // the source suspended. All header fields precede SOS-1, so this is
            // sufficient, and it keeps the parse input independent of entropy
            // content. `priv_state.source` still holds every drained byte for
            // the eventual decode; only this temporary parse copy is truncated.
            let end: usize = priv_state.body_scan_cursor.min(src_bytes.len());
            let mut v: Vec<u8> = Vec::with_capacity(end + 2);
            v.extend_from_slice(&src_bytes[..end]);
            v.extend_from_slice(&[0xFF, 0xD9]);
            v
        } else {
            Vec::new()
        };
        let bytes: &[u8] = if needs_synthetic_eoi {
            &parse_holder
        } else {
            src_bytes
        };

        let mut decoder: libjpeg_turbo_rs::Decoder<'_> = match libjpeg_turbo_rs::Decoder::new(bytes)
        {
            Ok(d) => d,
            Err(e) => {
                priv_state.last_error =
                    CString::new(format!("jpeg_read_header: {e}")).unwrap_or_default();
                // Distinguish "input is incomplete (need more data)" from
                // "input is syntactically complete but corrupt." Bytes ending
                // in EOI (FF D9) are syntactically complete, so a decoder
                // error means real corruption — invoke the consumer's
                // `error_exit` per the libjpeg.txt §3 contract so a
                // setjmp/longjmp handler can recover. Bytes without EOI may
                // still be truncated, so leave the existing JPEG_SUSPENDED
                // semantics intact and let the consumer feed more.
                //
                // The `bytes.len() >= 4` guard avoids a false positive on
                // tiny inputs whose entire content happens to look like
                // `FF D9` — a real JPEG always has SOI (FF D8) before EOI.
                let appears_complete: bool = bytes.len() >= 4
                    && bytes[bytes.len() - 2] == 0xFF
                    && bytes[bytes.len() - 1] == 0xD9;
                if appears_complete {
                    // Code 12 maps to JERR_BAD_LENGTH in the upstream v8
                    // alphabetical enum (jerror.h, with JPEG_LIB_VERSION=80
                    // excluding JERR_ARITH_NOTIMPL). It's the closest fit
                    // for the most common decoder rejection cause (bogus
                    // marker length); other causes still land here with the
                    // same code, and the consumer's `format_message` looks
                    // up the description in the message table. Consumers
                    // that care about the specific cause read
                    // `priv_state.last_error` (already populated above).
                    invoke_error_exit(cinfo, 12);
                    // If a custom handler returns instead of longjmping
                    // (which violates the libjpeg contract), fall through
                    // to JPEG_SUSPENDED so the caller still sees a
                    // non-success return.
                }
                return JPEG_SUSPENDED;
            }
        };
        // Re-parse with the per-cinfo marker save list so APP/COM markers
        // the caller asked for via `jpeg_save_markers` land in
        // `decoder.saved_markers()`. Stock libjpeg-turbo populates
        // `cinfo->marker_list` during `jpeg_read_header` before
        // `jpeg_read_coefficients` runs; downstream `transupp` callers
        // (stock `jpegtran -copy all <op>`) iterate that list to copy
        // ICC profiles and other APP markers across to dstinfo via
        // `jpeg_write_marker`. Without this re-parse the list stays empty
        // and `jcopy_markers_execute` finds nothing to forward, so the
        // ICC chunk on `monkey12.jpg` (and any APP/COM marker on other
        // fixtures) silently disappears in transcode output.
        let save_config: libjpeg_turbo_rs::MarkerSaveConfig =
            marker_save_to_config(&priv_state.marker_save);
        if !matches!(save_config, libjpeg_turbo_rs::MarkerSaveConfig::None) {
            decoder.save_markers(save_config);
        }

        let frame: &libjpeg_turbo_rs::FrameHeader = decoder.header();
        c.image_width = frame.width as JDimension;
        c.image_height = frame.height as JDimension;
        c.num_components = frame.components.len() as c_int;
        c.data_precision = frame.precision as c_int;
        c.progressive_mode = if frame.is_progressive { 1 } else { 0 };

        // Populate JFIF presence + density from the parsed APP0 marker.
        // Stock libjpeg-turbo sets `cinfo.saw_JFIF_marker`,
        // `JFIF_{major,minor}_version`, and
        // `density_unit / X_density / Y_density` during
        // `jpeg_read_header`; without this, `jpegtran -copy all <op>`
        // paths that go through
        // `jpeg_copy_critical_parameters → materialize_foreign_coef_arrays`
        // (no `jpeg_start_decompress` call) miss the source's JFIF
        // metadata and emit defaults — flagged at session-stop after
        // 2f6683b. The follow-up codex review (round-13) tightened the
        // requirement to use the actual APP0 presence and version,
        // not a density-based heuristic, so the parser surfaces both.
        let density: &libjpeg_turbo_rs::DensityInfo = decoder.density();
        let density_unit_raw: u8 = match density.unit {
            libjpeg_turbo_rs::DensityUnit::Unknown => 0,
            libjpeg_turbo_rs::DensityUnit::Dpi => 1,
            libjpeg_turbo_rs::DensityUnit::Dpcm => 2,
        };
        c.density_unit = density_unit_raw;
        c.X_density = density.x;
        c.Y_density = density.y;
        let saw_jfif: bool = decoder.saw_jfif_marker();
        let (jfif_major, jfif_minor): (u8, u8) = decoder.jfif_version();
        c.saw_JFIF_marker = if saw_jfif { 1 } else { 0 };
        // Reset to libjpeg's per-datastream JFIF version default `(1, 1)`
        // before optionally overwriting with the parsed APP0 values.
        // Stock libjpeg installs this default at every SOI, so a
        // decompressor reused across a JFIF 1.02 image followed by a
        // no-APP0 image must observe `(1, 1)` on the second pass —
        // codex round-15 review caught this stale-state hazard.
        c.JFIF_major_version = 1;
        c.JFIF_minor_version = 1;
        if saw_jfif {
            c.JFIF_major_version = jfif_major;
            c.JFIF_minor_version = jfif_minor;
        }
        // libjpeg's `is_baseline` flag: TRUE if SOF0 was encountered. We
        // approximate by clearing it for progressive/lossless streams.
        c.is_baseline = if !frame.is_progressive && !frame.is_lossless {
            1
        } else {
            0
        };
        // Populate `arith_code` from the parsed SOF marker type.
        // SOF9/SOF10/SOF11 → arithmetic (1); SOF0/SOF1/SOF2/SOF3 → Huffman (0).
        c.arith_code = if decoder.is_arithmetic() { 1 } else { 0 };

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

        // Build `c.marker_list` from the saved APP/COM markers so stock
        // C consumers (the most important being `transupp::jcopy_markers_execute`
        // invoked by `jpegtran -copy all`) can iterate the source's
        // markers exactly as upstream libjpeg-turbo allows. The byte
        // payload is moved out of `decoder.saved_markers()` and stashed
        // in `marker_list_storage` so the cinfo's `marker_list` pointer
        // remains valid until `jpeg_destroy_decompress` runs.
        //
        // Null `c.marker_list` *before* clearing the backing storage so
        // there is never an instant where the cinfo exposes a pointer to
        // freed memory — even though only this thread can observe the
        // window, defensive ordering keeps the invariant local and
        // obvious to a future reader.
        populate_marker_list(c, priv_state, decoder.saved_markers().to_vec());

        c.global_state = DSTATE_INHEADER;
        priv_state.last_error = CString::new("No error").expect("static");
        // Record that header parse completed so `jpeg_consume_input`
        // doesn't re-run the parser on subsequent polls and clobber
        // caller-set fields like `out_color_space` / `comp_info` /
        // `quantize_colors`.
        priv_state.header_parsed_ok = true;
        JPEG_HEADER_OK
    })
}

/// Rebuild `cinfo->marker_list` from the decoder's saved APP/COM markers,
/// honoring each marker's `jpeg_save_markers` `length_limit`. Backs the list
/// with `marker_list_storage` (boxed nodes, address-stable) and threads the
/// `next` pointers, so stock C consumers (e.g. `jcopy_markers_execute` from
/// `jpegtran -copy all`) can iterate them. Shared by `jpeg_read_header` (full
/// stream) and the P4-13 deferred decode (`ensure_decoded_deferred`), so a
/// suspending stream with markers after the first SOS preserves them too.
fn populate_marker_list(
    c: &mut JpegDecompressPublic,
    priv_state: &mut DecompressPrivate,
    saved: Vec<libjpeg_turbo_rs::SavedMarker>,
) {
    // Null `c.marker_list` *before* clearing the backing storage so the cinfo
    // never exposes a pointer into freed memory, even momentarily.
    c.marker_list = std::ptr::null_mut();
    priv_state.marker_list_storage.clear();
    for marker in saved {
        // `original_length` is the full body length; `data_length`/`data` are
        // truncated to `min(original_length, length_limit)` per the
        // `jpeg_save_markers` contract.
        let original_len: usize = marker.data.len();
        let limit: usize = priv_state
            .marker_save
            .limits
            .get(&marker.code)
            .copied()
            .map(|l| l as usize)
            .unwrap_or(usize::MAX);
        let truncated_len: usize = original_len.min(limit);
        let mut full: Vec<u8> = marker.data;
        full.truncate(truncated_len);
        let payload: Box<[u8]> = full.into_boxed_slice();
        priv_state.marker_list_storage.push(Box::new(OwnedMarker {
            public: JpegMarkerStructPublic {
                next: std::ptr::null_mut(),
                marker: marker.code,
                original_length: original_len as c_uint,
                data_length: truncated_len as c_uint,
                data: std::ptr::null_mut(),
            },
            payload,
        }));
    }
    for node in priv_state.marker_list_storage.iter_mut() {
        node.public.data = node.payload.as_mut_ptr();
    }
    let len: usize = priv_state.marker_list_storage.len();
    if len > 1 {
        let next_ptrs: Vec<*mut JpegMarkerStructPublic> = priv_state
            .marker_list_storage
            .iter_mut()
            .skip(1)
            .map(|n| &mut n.public as *mut _)
            .collect();
        for (i, node) in priv_state.marker_list_storage.iter_mut().enumerate() {
            if i < next_ptrs.len() {
                node.public.next = next_ptrs[i];
            }
        }
    }
    c.marker_list = priv_state
        .marker_list_storage
        .first_mut()
        .map(|n| &mut n.public as *mut _)
        .unwrap_or(std::ptr::null_mut());
}

/// Re-parse the now-complete `priv_state.source` for saved APP/COM markers and
/// rebuild `cinfo->marker_list` (P4-13). `jpeg_read_header` only parsed the
/// through-SOS prefix for a suspending source, so any markers between scans
/// were not yet visible; once the body drain reaches EOI, every completion path
/// (`jpeg_consume_input`, `finish_body_drain`) calls this so non-scanline
/// consumers see the same marker list the non-suspending path produces. No-op
/// unless `jpeg_save_markers` / a marker processor is active. Mirrors the marker
/// setup in `jpeg_read_header` (`Decoder::new` → `save_markers` →
/// `saved_markers`).
fn rebuild_marker_list_from_source(
    c: &mut JpegDecompressPublic,
    priv_state: &mut DecompressPrivate,
) {
    let save_config: libjpeg_turbo_rs::MarkerSaveConfig =
        marker_save_to_config(&priv_state.marker_save);
    let saving_enabled: bool = !matches!(save_config, libjpeg_turbo_rs::MarkerSaveConfig::None)
        || !priv_state.marker_processors.is_empty();
    if !saving_enabled {
        return;
    }
    let bytes: Vec<u8> = match priv_state.source.as_bytes() {
        Some(b) if b.len() >= 2 => b.to_vec(),
        _ => return,
    };
    let mut decoder: libjpeg_turbo_rs::Decoder<'_> = match libjpeg_turbo_rs::Decoder::new(&bytes) {
        Ok(d) => d,
        Err(_) => return,
    };
    if !matches!(save_config, libjpeg_turbo_rs::MarkerSaveConfig::None) {
        decoder.save_markers(save_config);
    }
    let saved: Vec<libjpeg_turbo_rs::SavedMarker> = decoder.saved_markers().to_vec();
    populate_marker_list(c, priv_state, saved);
}

// ---------------------------------------------------------------------------
// `jpeg_start_decompress` — subtask #6.
// ---------------------------------------------------------------------------

/// Map a `J_COLOR_SPACE` int to the `PixelFormat` the Rust decoder
/// should emit. Returns `None` for spaces we don't currently surface.
fn jcs_to_pixel_format(cs: c_int) -> Option<PixelFormat> {
    // J_COLOR_SPACE values per libjpeg-turbo's `jmorecfg.h`:
    // `JCS_EXT_RGB = 6` … `JCS_EXT_ARGB = 15`. Pillow's `_imaging.so`
    // sets `out_color_space = JCS_EXT_RGBX` (= 7) on decode so the
    // decoder writes 4-byte-stride RGB+padding straight into PIL's
    // RGBA-shaped frame buffer. Without these arms we'd map
    // `JCS_EXT_RGBX` to the fallback `PixelFormat::Rgb` (3
    // bytes/pixel) and copy 3-byte rows into PIL's 4-byte rows,
    // shifting every pixel by one channel and producing severely
    // distorted round-trip output.
    match cs {
        JCS_GRAYSCALE => Some(PixelFormat::Grayscale),
        JCS_RGB => Some(PixelFormat::Rgb),
        JCS_YCBCR => Some(PixelFormat::Rgb), // decoder converts YCbCr->RGB
        JCS_CMYK => Some(PixelFormat::Cmyk),
        // Extended color spaces (libjpeg-turbo only).
        6 /* JCS_EXT_RGB */ => Some(PixelFormat::Rgb),
        7 /* JCS_EXT_RGBX */ => Some(PixelFormat::Rgbx),
        8 /* JCS_EXT_BGR */ => Some(PixelFormat::Bgr),
        9 /* JCS_EXT_BGRX */ => Some(PixelFormat::Bgrx),
        10 /* JCS_EXT_XBGR */ => Some(PixelFormat::Xbgr),
        11 /* JCS_EXT_XRGB */ => Some(PixelFormat::Xrgb),
        12 /* JCS_EXT_RGBA */ => Some(PixelFormat::Rgba),
        13 /* JCS_EXT_BGRA */ => Some(PixelFormat::Bgra),
        14 /* JCS_EXT_ABGR */ => Some(PixelFormat::Abgr),
        15 /* JCS_EXT_ARGB */ => Some(PixelFormat::Argb),
        _ => None,
    }
}

/// Eagerly decode the image so `jpeg_read_scanlines` can serve rows out
/// of a backing buffer. Returns TRUE on success (libjpeg contract: never
/// returns FALSE except when suspending via a nonblocking source).
#[no_mangle]
pub extern "C" fn jpeg_start_decompress(cinfo: *mut c_void) -> CBoolean {
    crate::unwind_guard!(0, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return 0,
        };

        // P4-13: the body is still draining from a suspending source.
        if priv_state.body_incomplete {
            if c.buffered_image != 0 {
                // Buffered-image mode: the caller drives `jpeg_consume_input` to
                // drain the body before reading pixels. Publish output dimensions
                // from the parsed header and defer the (buffered) pixel decode
                // until the body is complete — `jpeg_read_scanlines` /
                // `jpeg_start_output` materialise `priv_state.decoded` lazily once
                // `jpeg_consume_input` has reached EOI.
                c.output_scanline = 0;
                c.global_state = DSTATE_SCANNING;
                let _ = c; // release the &mut before jpeg_calc_output_dimensions re-borrows
                jpeg_calc_output_dimensions(cinfo);
                return 1;
            }
            // Non-buffered mode with a suspending source: finish draining the
            // whole body to EOI now, resuming the live source manager. If it
            // suspends, report suspension (return FALSE) so the caller refills
            // and retries — matching upstream's multi-scan PRELOAD behaviour.
            if !finish_body_drain(c, priv_state) {
                return 0;
            }
            // Source is now complete; fall through to the normal decode path.
        }

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
        // which misroutes djpeg's precision-dispatched decode loop. The
        // actual decode happens lazily in `jpeg12_read_scanlines` /
        // `jpeg16_read_scanlines` / `jpeg12_read_raw_data`.
        //
        // Populate output dimensions inline so the 12-bit raw-data path
        // doesn't depend on the caller having invoked the optional
        // `jpeg_calc_output_dimensions` helper. The standard libjpeg
        // sequence — `jpeg_read_header`; tweak params; `jpeg_start_decompress`;
        // `jpeg12_read_raw_data` — leaves `output_width / output_height /
        // max_v_samp_factor / min_DCT_v_scaled_size` at zero unless we
        // populate them here, which would short-circuit the EOF check at
        // the top of `jpeg12_read_raw_data` (codex review of f0dc137).
        if c.data_precision > 8 {
            c.output_scanline = 0;
            c.global_state = DSTATE_SCANNING;
            // The `c: &mut JpegDecompressPublic` borrow is released
            // when this scope returns. `jpeg_calc_output_dimensions`
            // re-borrows `cinfo` via its own `cinfo_mut` call;
            // returning immediately after means the two mutable
            // borrows never overlap. Re-borrows through raw pointers
            // are safe because the public struct lives behind the
            // caller's allocation (not Rust-owned), so aliasing is the
            // unsafe-FFI contract, not a Rust UB.
            let _ = c;
            jpeg_calc_output_dimensions(cinfo);
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
            match run_decoder_for_start(&bytes, format, save_config, &priv_state.marker_processors)
            {
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

        // `jpeg_read_header` already populated `saw_JFIF_marker`,
        // `JFIF_{major,minor}_version`, and `density_unit / X_density /
        // Y_density` from the parsed APP0 marker (round-13 fix at
        // b435574+ codex follow-up). Re-asserting density from `image`
        // keeps a single source of truth in case a caller bypassed
        // `jpeg_read_header` and went straight to start_decompress —
        // an unusual path, but harmless here since `image.density` came
        // out of the same parser.
        let density: libjpeg_turbo_rs::DensityInfo = image.density;
        let density_unit_raw: u8 = match density.unit {
            libjpeg_turbo_rs::DensityUnit::Unknown => 0,
            libjpeg_turbo_rs::DensityUnit::Dpi => 1,
            libjpeg_turbo_rs::DensityUnit::Dpcm => 2,
        };
        c.density_unit = density_unit_raw;
        c.X_density = density.x;
        c.Y_density = density.y;

        priv_state.decoded = Some(image);
        priv_state.last_error = CString::new("No error").expect("static");
        1
    })
}

/// Build a [`MarkerSaveConfig`] from the set of per-code length limits
/// accumulated by `jpeg_save_markers`. A zero limit clears saving, so
/// we skip those entries when composing the final set.
///
/// Returns `None` if no markers are enabled. Returns `Specific(codes)`
/// otherwise. Per-marker body truncation is applied separately when building
/// `cinfo->marker_list` from `Image.saved_markers`, so the Rust library saves
/// the full marker body and the shim truncates to the requested `length_limit`.
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

/// Materialise the deferred 8-bit pixel decode for the P4-13 buffered-image
/// path: when `jpeg_start_decompress` deferred decoding because the body was
/// still draining, run the buffered decode now (on the complete source) and
/// populate the output fields exactly as `jpeg_start_decompress` would have.
/// Returns `false` if the body is not yet complete or the decode failed.
fn ensure_decoded_deferred(
    c: &mut JpegDecompressPublic,
    priv_state: &mut DecompressPrivate,
) -> bool {
    if priv_state.decoded.is_some() {
        return true;
    }
    if priv_state.body_incomplete || c.data_precision > 8 {
        return false;
    }
    let bytes: Vec<u8> = match priv_state.source.as_bytes() {
        Some(b) => b.to_vec(),
        None => return false,
    };
    let format: PixelFormat = jcs_to_pixel_format(c.out_color_space).unwrap_or(PixelFormat::Rgb);
    let save_config: libjpeg_turbo_rs::MarkerSaveConfig =
        marker_save_to_config(&priv_state.marker_save);
    match run_decoder_for_start(&bytes, format, save_config, &priv_state.marker_processors) {
        Ok(image) => {
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
            c.data_precision = image.precision as c_int;
            // P4-13: jpeg_read_header parsed only the through-SOS prefix, so
            // cinfo->marker_list holds only the pre-SOS markers. The deferred
            // full-stream decode saw any APP/COM markers that appear after the
            // first SOS too — rebuild the list from the complete set so the
            // suspending path preserves inter-scan metadata exactly like the
            // non-suspending path.
            populate_marker_list(c, priv_state, image.saved_markers.clone());
            priv_state.decoded = Some(image);
            true
        }
        Err(e) => {
            priv_state.last_error =
                CString::new(format!("jpeg_read_scanlines: {e}")).unwrap_or_default();
            false
        }
    }
}

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
    crate::unwind_guard!(0, {
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
        // P4-13: in buffered-image mode the pixel decode was deferred at
        // `jpeg_start_decompress` until `jpeg_consume_input` finished draining
        // the body. Materialise it now (once the body is complete).
        if priv_state.decoded.is_none() && !ensure_decoded_deferred(c, priv_state) {
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
    })
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
    crate::unwind_guard!((), {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };

        // No IDCT scaling: output dims = image dims after applying the
        // scale_num/scale_denom fraction (most callers leave 1/1).
        let scale_num: u64 = c.scale_num.max(1) as u64;
        let scale_denom: u64 = c.scale_denom.max(1) as u64;
        c.output_width = (c.image_width as u64 * scale_num).div_ceil(scale_denom) as JDimension;
        c.output_height = (c.image_height as u64 * scale_num).div_ceil(scale_denom) as JDimension;

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
                comp.downsampled_width = ((c.image_width as u64 * comp.h_samp_factor as u64)
                    .div_ceil(denom_w)
                    * 8) as JDimension;
                comp.downsampled_height = ((c.image_height as u64 * comp.v_samp_factor as u64)
                    .div_ceil(denom_h)
                    * 8) as JDimension;
            }
        }

        // out_color_components count per the J_COLOR_SPACE selected. Mirror
        // the rgb_pixelsize table used by jdmaster.c:341-365 so extended
        // color spaces (JCS_EXT_*) land on the correct channel count.
        // JCS_EXT_RGB=6, JCS_EXT_RGBX=7, … JCS_EXT_ARGB=15 per
        // libjpeg-turbo's `jmorecfg.h` enum order.
        c.out_color_components = match c.out_color_space {
            JCS_GRAYSCALE => 1,
            JCS_RGB | JCS_YCBCR => 3,
            JCS_CMYK | JCS_YCCK => 4,
            6 | 8 => 3,                               // JCS_EXT_RGB / JCS_EXT_BGR
            7 | 9 | 10 | 11 | 12 | 13 | 14 | 15 => 4, // *X / X* / *A / A*
            _ => c.num_components,
        };
        c.output_components = if c.quantize_colors != 0 {
            1
        } else {
            c.out_color_components
        };
        c.rec_outbuf_height = 1;
    })
}

/// Free everything the cinfo's memory manager allocated under
/// `JPOOL_IMAGE`, i.e. the `jvirt_barray_ptr*` array slot, the
/// per-component `JVirtBarrayControl`s, and the populated block rows
/// that `jpeg_read_coefficients` (or transupp's
/// `jtransform_request_workspace`) requested.
///
/// Mirrors what stock libjpeg's `jpeg_abort` does for the decompress
/// side: walk the lifetime classes that aren't `JPOOL_PERMANENT` and
/// release them. We only release `JPOOL_IMAGE` here because that's
/// the only pool the decompress side touches in our shim today.
fn release_decompress_image_pool(cinfo: *mut c_void, mem_ptr: *mut c_void) {
    if mem_ptr.is_null() {
        return;
    }
    let mem: &memmgr::JpegMemoryMgr = unsafe { &*(mem_ptr as *const memmgr::JpegMemoryMgr) };
    if let Some(free_pool) = mem.free_pool {
        unsafe { free_pool(cinfo, memmgr::JPOOL_IMAGE) };
    }
}

/// Close out the decode pass. Returns TRUE unless suspended (which we
/// never do — the entire stream is present in memory).
#[no_mangle]
pub extern "C" fn jpeg_finish_decompress(cinfo: *mut c_void) -> CBoolean {
    crate::unwind_guard!(0, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        c.global_state = DSTATE_STOPPING;
        // Drop the decoded image and any source bytes bridged from a
        // caller-installed `cinfo->src` so the next `jpeg_read_header` on
        // this handle re-runs the bridge against whatever new source the
        // caller installs. Without dropping `source`, a libjpeg-style
        // caller that reuses the same decompressor with a *different*
        // direct source manager would silently re-decode the previous
        // image's bytes.
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        if let Some(priv_state) = unsafe { priv_from_ptr(priv_ptr) } {
            priv_state.decoded = None;
            priv_state.source = JpegSource::None;
            priv_state.bridge_partial.clear();
            // A reuse path will re-call `jpeg_read_header` for the new
            // image; clear the flag so `jpeg_consume_input` doesn't
            // short-circuit.
            priv_state.header_parsed_ok = false;
            // P4-13 incremental-input state, reset for handle reuse.
            priv_state.body_incomplete = false;
            priv_state.body_scan_cursor = 0;
            priv_state.eoi_seen = false;
            // Reset the public scan counter too — the suspending path advances
            // it per SOS, and a reused handle decoding a new (e.g. mem_src)
            // image must not expose the previous image's scan count.
            c.input_scan_number = 0;
            // Drop the previous-image's parsed coefficients and unhook
            // the foreign `jvirt_barray_ptr*` from the global side
            // table. Without this, a `finish_decompress` → new
            // `mem_src` → new `read_header` → `read_coefficients` reuse
            // would short-circuit on the stale `coef_array_ptr` and
            // hand the second JPEG's caller the *first* JPEG's barrays
            // — caught by codex review of 809f52a.
            //
            // Also release the JPOOL_IMAGE allocations that backed the
            // old barray array (the `*mut JVirtBarrayControl` slots,
            // the per-component `JVirtBarrayControl`s, and the
            // populated block rows). Upstream `jpeg_finish_decompress`
            // reaches the same end state via `jpeg_abort` →
            // `free_pool(JPOOL_IMAGE)`; without this the next
            // `read_coefficients` on the same handle would re-allocate
            // a parallel set without ever freeing the first one,
            // ballooning memory across finish/reuse cycles — codex
            // round-8 review of b7f690d.
            if !priv_state.coef_array_ptr.is_null() {
                coef_unregister_array(priv_state.coef_array_ptr as *const c_void);
                priv_state.coef_array_ptr = std::ptr::null_mut();
                release_decompress_image_pool(cinfo, c.mem);
            }
            priv_state.coefficients = None;
            // Drop the previous-image saved-marker list so a reuse of
            // this cinfo for a different image cannot expose the old
            // image's APP/COM markers via `c.marker_list`. Stock
            // libjpeg-turbo reaches the same end state through `jpeg_abort
            // → free_pool(JPOOL_IMAGE)` in its finish path.
            c.marker_list = std::ptr::null_mut();
            priv_state.marker_list_storage.clear();
            // Drop the lazy raw-data caches so a reuse of this cinfo for
            // a different JPEG re-runs `decompress_raw` /
            // `decompress_raw_12` against the new source. Without these,
            // the `is_none()` guards at the top of `jpeg_read_raw_data` /
            // `jpeg12_read_raw_data` would short-circuit on the previous
            // image's planes and copy stale rows — codex review of f0dc137
            // flagged the 12-bit half; the 8-bit cache had the same latent
            // bug, so both are cleared here.
            priv_state.raw_image_cache = None;
            priv_state.raw_image_cache_12 = None;
            priv_state.raw_rows_consumed.clear();
        }
        1
    })
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
    crate::unwind_guard!((), {
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
    })
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_out_cs(cinfo: *mut c_void, cs: c_int) {
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_mut(cinfo) } {
            c.out_color_space = cs;
        }
    })
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_density_unit(cinfo: *mut c_void) -> c_int {
    crate::unwind_guard!(-1, {
        match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c.density_unit as c_int,
            None => -1,
        }
    })
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_x_density(cinfo: *mut c_void) -> c_int {
    crate::unwind_guard!(-1, {
        match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c.X_density as c_int,
            None => -1,
        }
    })
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_y_density(cinfo: *mut c_void) -> c_int {
    crate::unwind_guard!(-1, {
        match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c.Y_density as c_int,
            None => -1,
        }
    })
}

/// Read `cinfo->arith_code` after `jpeg_read_header`. Returns 1 for
/// arithmetic-coded streams, 0 for Huffman, -1 if `cinfo` is null.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_arith_code(cinfo: *mut c_void) -> c_int {
    crate::unwind_guard!(-1, {
        match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c.arith_code as c_int,
            None => -1,
        }
    })
}

/// Return `cinfo->marker_list` — the head of the saved-marker linked list
/// populated by `jpeg_read_header`. Tests use this to inspect the saved
/// bodies without hard-coding struct offsets.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_marker_list(cinfo: *mut c_void) -> *mut JpegMarkerStructPublic {
    crate::unwind_guard!(std::ptr::null_mut(), {
        match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c.marker_list,
            None => std::ptr::null_mut(),
        }
    })
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_output_dims(
    cinfo: *mut c_void,
    out_w: *mut u32,
    out_h: *mut u32,
    out_components: *mut c_int,
    out_cs: *mut c_int,
) {
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!(0, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        let total: JDimension = c.output_height;
        let remaining: JDimension = total.saturating_sub(c.output_scanline);
        let skip: JDimension = std::cmp::min(num_lines, remaining);
        c.output_scanline = c.output_scanline.saturating_add(skip);
        skip
    })
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
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
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
            // Stock libjpeg-turbo's `jpeg_save_markers` (jdmarker.c) raises
            // a non-zero `length_limit` to the minimum needed to identify
            // JFIF APP0 (14 bytes) or Adobe APP14 (12 bytes) — the markers
            // libjpeg's own machinery relies on for colorspace
            // classification. Apply the same floor so a caller that
            // requested e.g. `jpeg_save_markers(cinfo, JPEG_APP0, 1)` sees
            // the canonical 14-byte minimum (matches the C contract; codex
            // review of the marker-list landing flagged the divergence).
            const APP0_DATA_LEN: c_uint = 14;
            const APP14_DATA_LEN: c_uint = 12;
            let effective: c_uint = if code == 0xE0 {
                length_limit.max(APP0_DATA_LEN)
            } else if code == 0xEE {
                length_limit.max(APP14_DATA_LEN)
            } else {
                length_limit
            };
            priv_state.marker_save.limits.insert(code, effective);
        }
    })
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
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!(0, {
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
    })
}

/// `jpeg_read_coefficients(cinfo) -> jvirt_barray_ptr *`.
///
/// Parses the input JPEG entropy-coded data to recover quantized DCT
/// coefficients without performing IDCT or color conversion. Returns
/// a real `jvirt_barray_ptr *` (an array of N component
/// `jvirt_barray_ptr`s allocated through `cinfo->mem`), populated with
/// the parsed coefficients in iMCU-aligned blocks. Each entry is a
/// `JVirtBarrayControl *` realised by `realize_virt_arrays`, so stock
/// `transupp` / `jtransform_*` helpers can iterate it via
/// `cinfo->mem->access_virt_barray` exactly as they would with
/// upstream libjpeg-turbo.
///
/// Storage lifetime is the cinfo's `JPOOL_IMAGE`: the array stays
/// valid until `jpeg_destroy_decompress` (or `jpeg_finish_decompress`
/// + reuse) frees the pool. Callers MUST NOT `free()` the pointer.
///
/// In-process consumers that immediately pass this pointer to
/// `jpeg_write_coefficients(dstinfo, …)` will hit a fast path that
/// recovers the parsed `JpegCoefficients` from the global side table
/// and skips the per-barray re-read; foreign arrays produced by
/// `jtransform_adjust_parameters` (i.e. the destination workspace
/// allocated by transupp) fall through to the slower
/// "materialise from barrays + cinfo metadata" path implemented in
/// `run_coefficient_writer_and_flush`.
#[no_mangle]
pub extern "C" fn jpeg_read_coefficients(cinfo: *mut c_void) -> *mut c_void {
    crate::unwind_guard!(std::ptr::null_mut(), {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return std::ptr::null_mut(),
        };
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return std::ptr::null_mut(),
        };

        // Fast-path: a previous successful `jpeg_read_coefficients` already
        // parsed and registered the array — return the same pointer so
        // repeated polls observe identical handles (matches libjpeg's
        // contract that the returned virt_barray array is stable across
        // reads on the same cinfo).
        if !priv_state.coef_array_ptr.is_null() && priv_state.coefficients.is_some() {
            return priv_state.coef_array_ptr;
        }

        // P4-13: if the header was parsed from a through-SOS prefix of a
        // suspending source, finish draining the body to EOI before parsing
        // coefficients (upstream `jpeg_read_coefficients` consumes to EOI). On
        // suspension, return null so the caller refills and retries.
        if priv_state.body_incomplete && !finish_body_drain(c, priv_state) {
            return std::ptr::null_mut();
        }

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

        // Stash the parsed coefficients in a CoefHandle so callers that
        // round-trip through this shim (read_coefficients →
        // write_coefficients on a sibling cinfo) get the cached
        // JpegCoefficients via the side table without rebuilding from
        // individual barray reads.
        priv_state.coefficients = Some(Box::new(CoefHandle {
            magic: CoefHandle::MAGIC,
            inner: coeffs,
        }));
        let handle_ptr: *const CoefHandle =
            priv_state.coefficients.as_deref().expect("just inserted") as *const CoefHandle;

        // Build the foreign-style `jvirt_barray_ptr*` array via cinfo->mem
        // so `transupp` (compiled into stock `jpegtran`) can index it as
        // a plain C array.
        let mem_ptr: *mut c_void = c.mem;
        if mem_ptr.is_null() {
            priv_state.last_error =
                CString::new("jpeg_read_coefficients: cinfo->mem is null").unwrap_or_default();
            return std::ptr::null_mut();
        }
        let mem: &memmgr::JpegMemoryMgr = unsafe { &*(mem_ptr as *const memmgr::JpegMemoryMgr) };
        let alloc_small: memmgr::AllocFn = match mem.alloc_small {
            Some(f) => f,
            None => {
                priv_state.last_error =
                    CString::new("jpeg_read_coefficients: alloc_small not wired")
                        .unwrap_or_default();
                return std::ptr::null_mut();
            }
        };
        let request_virt_barray: memmgr::RequestVirtBarrayFn = match mem.request_virt_barray {
            Some(f) => f,
            None => {
                priv_state.last_error =
                    CString::new("jpeg_read_coefficients: request_virt_barray not wired")
                        .unwrap_or_default();
                return std::ptr::null_mut();
            }
        };
        let realize_virt_arrays: memmgr::RealizeVirtArraysFn = match mem.realize_virt_arrays {
            Some(f) => f,
            None => {
                priv_state.last_error =
                    CString::new("jpeg_read_coefficients: realize_virt_arrays not wired")
                        .unwrap_or_default();
                return std::ptr::null_mut();
            }
        };
        let access_virt_barray: memmgr::AccessVirtBarrayFn = match mem.access_virt_barray {
            Some(f) => f,
            None => {
                priv_state.last_error =
                    CString::new("jpeg_read_coefficients: access_virt_barray not wired")
                        .unwrap_or_default();
                return std::ptr::null_mut();
            }
        };

        let inner: &libjpeg_turbo_rs::JpegCoefficients =
            &priv_state.coefficients.as_deref().expect("set above").inner;
        let n: usize = inner.components.len();
        if n == 0 {
            return std::ptr::null_mut();
        }

        // Allocate the array of `jvirt_barray_ptr` slots from JPOOL_IMAGE.
        let array_bytes: usize = n * std::mem::size_of::<*mut memmgr::JVirtBarrayControl>();
        let array_raw: *mut c_void =
            unsafe { alloc_small(cinfo, memmgr::JPOOL_IMAGE, array_bytes) };
        if array_raw.is_null() {
            priv_state.last_error =
                CString::new("jpeg_read_coefficients: alloc_small returned null")
                    .unwrap_or_default();
            return std::ptr::null_mut();
        }
        let array_slot: *mut *mut memmgr::JVirtBarrayControl =
            array_raw as *mut *mut memmgr::JVirtBarrayControl;

        // Allocate per-component virt barrays. `comp.blocks_x / blocks_y`
        // are already iMCU-padded by the parser (see `read_coefficients`
        // in `src/api/coefficient.rs`), so we can pass them straight
        // through as the requested barray dimensions.
        for ci in 0..n {
            let comp: &libjpeg_turbo_rs::ComponentCoefficients = &inner.components[ci];
            let blocks_x: JDimension = comp.blocks_x as JDimension;
            let blocks_y: JDimension = comp.blocks_y as JDimension;
            let v_samp: JDimension = comp.v_sampling as JDimension;
            let barray: *mut memmgr::JVirtBarrayControl = unsafe {
                request_virt_barray(
                    cinfo,
                    memmgr::JPOOL_IMAGE,
                    /*pre_zero=*/ 0,
                    blocks_x,
                    blocks_y,
                    v_samp.max(1),
                )
            };
            if barray.is_null() {
                priv_state.last_error = CString::new(format!(
                    "jpeg_read_coefficients: request_virt_barray(comp={ci}) returned null"
                ))
                .unwrap_or_default();
                return std::ptr::null_mut();
            }
            // SAFETY: `array_slot` points at an `n`-element array allocated
            // above; `ci < n`.
            unsafe {
                *array_slot.add(ci) = barray;
            }
        }

        // Realize backing storage for every requested virt array.
        unsafe { realize_virt_arrays(cinfo) };

        // Populate each barray with the parsed coefficient blocks, copying
        // through `access_virt_barray` so we use the same access pattern
        // a foreign caller would.
        for ci in 0..n {
            let comp: &libjpeg_turbo_rs::ComponentCoefficients = &inner.components[ci];
            let blocks_x_u: usize = comp.blocks_x;
            let blocks_y_u: usize = comp.blocks_y;
            let blocks_x: JDimension = blocks_x_u as JDimension;
            let blocks_y: JDimension = blocks_y_u as JDimension;
            let barray: *mut memmgr::JVirtBarrayControl = unsafe { *array_slot.add(ci) };
            let row_array: memmgr::JBlockArray = unsafe {
                access_virt_barray(cinfo, barray, 0, blocks_y, /*writable=*/ 1)
            };
            if row_array.is_null() {
                priv_state.last_error = CString::new(format!(
                    "jpeg_read_coefficients: access_virt_barray(comp={ci}) returned null"
                ))
                .unwrap_or_default();
                return std::ptr::null_mut();
            }
            for r in 0..blocks_y_u {
                // SAFETY: row_array has `blocks_y` row pointers, each
                // pointing to `blocks_x` JBlocks, allocated by
                // `alloc_barray_impl`.
                let row_ptr: memmgr::JBlockRow = unsafe { *row_array.add(r) };
                let row_blocks: &mut [memmgr::JBlock] =
                    unsafe { std::slice::from_raw_parts_mut(row_ptr, blocks_x_u) };
                for (c_idx, dst_block) in row_blocks.iter_mut().enumerate() {
                    let blk_idx: usize = r * blocks_x_u + c_idx;
                    // Convert zigzag → natural row-major so foreign
                    // consumers (stock `transupp::do_rot_*` /
                    // `do_transpose / …`) can index coefficients as
                    // `block[i*8+j]` representing the (i,j) DCT element.
                    // Our parser emits zigzag (the order DQT stores), so
                    // we un-zigzag here. `materialize_foreign_coef_arrays`
                    // re-zigzags before re-encoding.
                    let zigzag_block: &[i16; 64] = &comp.blocks[blk_idx];
                    for (natural_pos, slot) in dst_block.iter_mut().enumerate() {
                        let zigzag_pos: usize =
                            libjpeg_turbo_rs::common::quant_table::NATURAL_ORDER[natural_pos];
                        *slot = zigzag_block[zigzag_pos];
                    }
                }
                let _ = blocks_x; // silence unused warning if width is 0.
            }
        }

        // Register the array → CoefHandle mapping so an in-process
        // `jpeg_write_coefficients` can shortcut to the cached
        // JpegCoefficients.
        coef_register_array(array_raw as *const c_void, handle_ptr);
        priv_state.coef_array_ptr = array_raw;

        array_raw
    })
}

/// `jpeg_copy_critical_parameters(srcinfo, dstinfo)`.
///
/// Copies the subset of fields needed by `jpegtran` to re-encode the
/// coefficient arrays returned from `jpeg_read_coefficients`. Mirrors
/// upstream libjpeg-turbo's `jctrans.c::jpeg_copy_critical_parameters`:
/// dimensions, color-space classification, sampling factors, quant
/// tables, JFIF version, density.
///
/// On invalid input (NULL pointers, header-not-yet-parsed, mismatched
/// `is_decompressor` flags) this is a tolerant no-op so callers that
/// chain it before establishing both cinfos cannot crash the process.
#[no_mangle]
pub extern "C" fn jpeg_copy_critical_parameters(srcinfo: *mut c_void, dstinfo: *mut c_void) {
    crate::unwind_guard!((), {
        if srcinfo.is_null() || dstinfo.is_null() {
            return;
        }
        // Snapshot every field we need from src before we borrow dst. The
        // dst borrow path will install a new memory manager pool entry
        // (jpeg_alloc_quant_table) that can outlive the snapshot, so a
        // single read here keeps lifetimes straight.
        let snapshot: SrcCriticalSnapshot = match unsafe { snapshot_src_for_copy(srcinfo) } {
            Some(s) => s,
            None => return,
        };

        let dst: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(dstinfo) } {
            Some(c) => c,
            None => return,
        };
        // Defensive heuristic to keep the existing
        // `*_copy_params_is_noop_safe` smoke test honest: a properly
        // initialised compress cinfo (post `jpeg_CreateCompress`) sets
        // `is_decompressor=0` AND has a non-NULL `mem` (the memory
        // manager installed in CreateCompress). A stack-allocated zeroed
        // dummy buffer fails the `mem` check; bail before
        // `jpeg_set_defaults` writes fields past the buffer's end.
        if dst.is_decompressor != 0 || dst.mem.is_null() {
            return;
        }

        // Step 1: fundamental image dims + input description.
        dst.image_width = snapshot.image_width;
        dst.image_height = snapshot.image_height;
        dst.input_components = snapshot.num_components;
        dst.in_color_space = snapshot.jpeg_color_space;
        // libjpeg 8+: `jpeg_width / jpeg_height` mirror `output_width /
        // output_height` when scaling is in effect; with default 1:1 the
        // output dims equal the image dims.
        dst.jpeg_width = snapshot.image_width;
        dst.jpeg_height = snapshot.image_height;

        // Step 2: install canonical defaults, then override colorspace so
        // the comp_info / quant_tbl_no allocation matches the source.
        jpeg_set_defaults(dstinfo);
        jpeg_set_colorspace(dstinfo, snapshot.jpeg_color_space);

        // Re-borrow because `jpeg_set_*` re-acquired the cinfo through the
        // pointer. The Rust borrow checker cannot see that.
        let dst: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(dstinfo) } {
            Some(c) => c,
            None => return,
        };

        dst.data_precision = snapshot.data_precision;
        dst.CCIR601_sampling = snapshot.ccir601_sampling;

        // Step 3: copy quant tables — allocate slots through
        // `jpeg_alloc_quant_table` so lifetime tracking matches stock
        // libjpeg (the slot lives in the cinfo's permanent pool).
        for tblno in 0..NUM_QUANT_TBLS {
            let src_qt: Option<[u16; DCTSIZE2]> = snapshot.quant_tables[tblno];
            let Some(src_quantval) = src_qt else { continue };
            let mut slot: *mut JQuantTblPublic = dst.quant_tbl_ptrs[tblno];
            if slot.is_null() {
                slot = jpeg_alloc_quant_table(dstinfo);
                if slot.is_null() {
                    return;
                }
                // Re-borrow once more to install the slot.
                let dst2: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(dstinfo) } {
                    Some(c) => c,
                    None => return,
                };
                dst2.quant_tbl_ptrs[tblno] = slot;
            }
            // SAFETY: `slot` was either allocated by us above or was already
            // installed by `jpeg_set_defaults`. Either way, it points at a
            // live `JQuantTblPublic` aligned for direct field writes.
            unsafe {
                (*slot).quantval = src_quantval;
                (*slot).sent_table = 0;
            }
        }

        // Step 4: copy num_components and per-component info. Re-borrow
        // because the quant-table install used a fresh borrow.
        let dst: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(dstinfo) } {
            Some(c) => c,
            None => return,
        };
        let n: usize = snapshot.num_components.max(0) as usize;
        if n == 0 || n > MAX_COMPS_OWNED {
            return;
        }
        dst.num_components = snapshot.num_components;
        if !dst.comp_info.is_null() {
            let dst_comps: &mut [JpegComponentInfoPublic] =
                unsafe { std::slice::from_raw_parts_mut(dst.comp_info, n) };
            for (ci, dst_comp) in dst_comps.iter_mut().enumerate().take(n) {
                let src_comp: &SrcComponentSnapshot = &snapshot.components[ci];
                dst_comp.component_id = src_comp.component_id;
                dst_comp.component_index = ci as c_int;
                dst_comp.h_samp_factor = src_comp.h_samp_factor;
                dst_comp.v_samp_factor = src_comp.v_samp_factor;
                dst_comp.quant_tbl_no = src_comp.quant_tbl_no;
                // Re-link the per-component quant_table pointer at the
                // newly-installed dst slot so downstream `jpeg_finish_compress`
                // (and any tooling that walks comp_info[i].quant_table) sees
                // the dst-side table, not a stray src pointer.
                let qno: usize = src_comp.quant_tbl_no.max(0) as usize;
                if qno < NUM_QUANT_TBLS {
                    dst_comp.quant_table = dst.quant_tbl_ptrs[qno];
                }
            }
        }

        // Step 5: JFIF version + density. Mirror upstream's "only copy on
        // saw_JFIF_marker, and not for major-version > 1" guard.
        if snapshot.saw_jfif_marker != 0 {
            if snapshot.jfif_major_version == 1 {
                dst.JFIF_major_version = snapshot.jfif_major_version;
                dst.JFIF_minor_version = snapshot.jfif_minor_version;
            }
            dst.density_unit = snapshot.density_unit;
            dst.X_density = snapshot.x_density;
            dst.Y_density = snapshot.y_density;
        }
    })
}

/// Per-component subset of the source snapshot used by
/// `jpeg_copy_critical_parameters`.
struct SrcComponentSnapshot {
    component_id: c_int,
    h_samp_factor: c_int,
    v_samp_factor: c_int,
    quant_tbl_no: c_int,
}

/// Snapshot of every src field `jpeg_copy_critical_parameters` reads,
/// so we can release the src borrow before we start mutating dst.
struct SrcCriticalSnapshot {
    image_width: JDimension,
    image_height: JDimension,
    num_components: c_int,
    jpeg_color_space: c_int,
    data_precision: c_int,
    ccir601_sampling: CBoolean,
    saw_jfif_marker: CBoolean,
    jfif_major_version: u8,
    jfif_minor_version: u8,
    density_unit: u8,
    x_density: u16,
    y_density: u16,
    quant_tables: [Option<[u16; DCTSIZE2]>; NUM_QUANT_TBLS],
    components: Vec<SrcComponentSnapshot>,
}

/// Read every field `jpeg_copy_critical_parameters` needs from the
/// source decompress cinfo into a self-owned struct. Returns `None`
/// for invalid input so the caller can no-op.
///
/// # Safety
/// `srcinfo` must be a valid `*mut JpegDecompressPublic` whose header
/// has been parsed (i.e. `num_components > 0`, `comp_info` non-NULL).
unsafe fn snapshot_src_for_copy(srcinfo: *mut c_void) -> Option<SrcCriticalSnapshot> {
    let src: &mut JpegDecompressPublic = unsafe { cinfo_mut(srcinfo) }?;
    if src.num_components <= 0 {
        return None;
    }
    let n: usize = src.num_components as usize;
    if src.comp_info.is_null() {
        return None;
    }
    let src_comps: &[JpegComponentInfoPublic] =
        unsafe { std::slice::from_raw_parts(src.comp_info, n) };
    let components: Vec<SrcComponentSnapshot> = src_comps
        .iter()
        .map(|c| SrcComponentSnapshot {
            component_id: c.component_id,
            h_samp_factor: c.h_samp_factor,
            v_samp_factor: c.v_samp_factor,
            quant_tbl_no: c.quant_tbl_no,
        })
        .collect();

    let mut quant_tables: [Option<[u16; DCTSIZE2]>; NUM_QUANT_TBLS] = Default::default();
    for (i, slot) in src.quant_tbl_ptrs.iter().enumerate() {
        if !slot.is_null() {
            // SAFETY: caller (header-parsed cinfo) ensures the slot
            // contains a live `JQuantTblPublic`.
            let qt: &JQuantTblPublic = unsafe { &**slot };
            quant_tables[i] = Some(qt.quantval);
        }
    }
    // Fallback: our `jpeg_read_header` does not yet populate the
    // public `quant_tbl_ptrs` slots (the parser keeps tables internally
    // until `read_coefficients` runs). When a caller has already called
    // `jpeg_read_coefficients`, the parsed `JpegCoefficients.quant_tables`
    // hold the canonical zigzag values — pull from there so jpegtran-style
    // sequences (`read_header → read_coefficients → copy_critical_parameters`)
    // see non-zero quant tables on the dst cinfo.
    let priv_ptr: *mut c_void = decompress_private_raw(srcinfo);
    if let Some(p) = unsafe { priv_from_ptr(priv_ptr) } {
        if let Some(handle) = p.coefficients.as_deref() {
            for (tblno, table) in handle.inner.quant_tables.iter().enumerate() {
                if tblno < NUM_QUANT_TBLS && quant_tables[tblno].is_none() {
                    quant_tables[tblno] = Some(*table);
                }
            }
        }
    }

    Some(SrcCriticalSnapshot {
        image_width: src.image_width,
        image_height: src.image_height,
        num_components: src.num_components,
        jpeg_color_space: src.jpeg_color_space,
        data_precision: src.data_precision,
        ccir601_sampling: src.CCIR601_sampling,
        saw_jfif_marker: src.saw_JFIF_marker,
        jfif_major_version: src.JFIF_major_version,
        jfif_minor_version: src.JFIF_minor_version,
        density_unit: src.density_unit,
        x_density: src.X_density,
        y_density: src.Y_density,
        quant_tables,
        components,
    })
}

/// `jpeg_core_output_dimensions(cinfo)`.
///
/// Mirrors upstream's "core" pre-crop output dimension computation,
/// which transupp / `jtransform_request_workspace` rely on to size
/// the destination workspace for `jpegtran -rotate / -transpose / …`.
/// We forward to `jpeg_calc_output_dimensions` because our shim does
/// not model a separate pre-crop path (cropping is applied at
/// scanline emission time), and that function already populates the
/// per-component `width_in_blocks / height_in_blocks /
/// downsampled_*` fields plus `max_h_samp_factor /
/// max_v_samp_factor / min_DCT_*_scaled_size` that transupp reads via
/// the `_min_DCT_*` accessors.
#[no_mangle]
pub extern "C" fn jpeg_core_output_dimensions(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        jpeg_calc_output_dimensions(cinfo);
    })
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
    crate::unwind_guard!(0, {
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
    })
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
    crate::unwind_guard!(0, {
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
    })
}

/// `jpeg12_crop_scanline(cinfo, *xoffset, *width)`.
#[no_mangle]
pub extern "C" fn jpeg12_crop_scanline(
    cinfo: *mut c_void,
    xoffset: *mut JDimension,
    width: *mut JDimension,
) {
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!(0, {
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
    })
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
// Raw-data decode entry points.
//
// Stock libjpeg's `jpeg_read_raw_data` delivers one iMCU row of
// pre-downsampled component planes per call, bypassing colour
// conversion and chroma upsampling.  The caller allocates a JSAMPIMAGE
// (pointer-to-pointer-to-pointer) with one entry per component; each
// component entry is an array of row pointers sized to hold
// `comp_v_samp_factor * DCTSIZE` rows of `plane_width` samples.
//
// Upstream contract (jdapistd.c `_jpeg_read_raw_data`):
//   - Returns `max_v_samp_factor * DCTSIZE` rows on success.
//   - Returns 0 with `JWRN_TOO_MUCH_DATA` if output_scanline >=
//     output_height (caller has already consumed all data).
//   - Errors: JERR_BUFFER_SIZE if max_lines is too small.
//
// Implementation — lazy materialisation:
//   On the first call, `decompress_raw` is invoked once on the stored
//   source bytes and the result is cached in `raw_image_cache`. Each
//   subsequent call copies the next block of rows from the cache into
//   the caller's row-pointer array and advances `output_scanline`.
//
// Scope limitations (8-bit baseline/progressive only):
//   - 12-bit raw-data is not implemented; `jpeg12_read_raw_data`
//     returns an error. Callers that only resolve the symbol at load
//     time (e.g. Pillow's libtiff dependency) are unaffected.
//   - Lossless (SOF3/SOF11) is not implemented; those streams do not
//     use iMCU rows in the DCT sense. Bail out with an error.
// ---------------------------------------------------------------------------

const DCTSIZE: usize = 8;

/// `jpeg_read_raw_data(cinfo, data, max_lines) -> JDIMENSION`.
///
/// Delivers one iMCU row of raw component-plane data per call.
/// Supports 8-bit baseline and progressive JPEGs only. Returns 0 and
/// sets `last_error` for 12-bit streams, lossless streams, or when
/// `max_lines` is too small for one full iMCU row.
#[no_mangle]
pub extern "C" fn jpeg_read_raw_data(
    cinfo: *mut c_void,
    data: *mut *mut *mut u8,
    max_lines: JDimension,
) -> JDimension {
    crate::unwind_guard!(0, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return 0,
        };

        // Reject unsupported precision (12-bit, 16-bit).
        if c.data_precision > 8 {
            priv_state.last_error = CString::new(
            "jpeg_read_raw_data: only 8-bit precision is supported; use jpeg12_read_raw_data for 12-bit"
        ).unwrap_or_default();
            return 0;
        }

        // P4-13: a suspending source's body may still be draining (normally
        // `jpeg_start_decompress` finished it, but a caller that reaches here
        // with `body_incomplete` must complete the drain before the raw cache
        // is built from `priv_state.source`). On suspension, return 0.
        if priv_state.body_incomplete && !finish_body_drain(c, priv_state) {
            return 0;
        }

        // EOF sentinel: output_scanline >= output_height → return 0 (no error,
        // matches libjpeg's JWRN_TOO_MUCH_DATA path).
        if c.output_scanline >= c.output_height {
            return 0;
        }

        // Validate caller supplied a non-null array pointer.
        if data.is_null() {
            priv_state.last_error =
                CString::new("jpeg_read_raw_data: data pointer is NULL").unwrap_or_default();
            return 0;
        }

        // `max_v_samp_factor` and per-component `v_samp_factor` are set by
        // `jpeg_read_header` / `jpeg_calc_output_dimensions`.
        let max_vsf: usize = c.max_v_samp_factor as usize;
        // For typical DCT-based streams, DCTSIZE = 8.  The public struct
        // exposes `min_DCT_v_scaled_size` (set to 8 in jpeg_calc_output_dimensions);
        // use it when positive, fall back to DCTSIZE otherwise.
        let dct_size: usize = if c.min_DCT_v_scaled_size > 0 {
            c.min_DCT_v_scaled_size as usize
        } else {
            DCTSIZE
        };
        let rows_per_imcu: usize = max_vsf * dct_size;

        // Validate buffer size: caller must supply at least `rows_per_imcu` rows.
        if (max_lines as usize) < rows_per_imcu {
            priv_state.last_error = CString::new(format!(
            "jpeg_read_raw_data: JERR_BUFFER_SIZE — max_lines ({max_lines}) < rows_per_iMCU ({rows_per_imcu})"
        )).unwrap_or_default();
            return 0;
        }

        // Lazy materialisation: decode the whole stream on the first call.
        if priv_state.raw_image_cache.is_none() {
            let bytes: &[u8] = match priv_state.source.as_bytes() {
                Some(b) => b,
                None => {
                    priv_state.last_error =
                        CString::new("jpeg_read_raw_data: no source").unwrap_or_default();
                    return 0;
                }
            };
            match libjpeg_turbo_rs::decompress_raw(bytes) {
                Ok(raw) => {
                    let ncomp: usize = raw.num_components;
                    priv_state.raw_rows_consumed = vec![0usize; ncomp];
                    priv_state.raw_image_cache = Some(raw);
                }
                Err(e) => {
                    priv_state.last_error =
                        CString::new(format!("jpeg_read_raw_data: decompress_raw failed: {e}"))
                            .unwrap_or_default();
                    return 0;
                }
            }
        }

        let raw: &libjpeg_turbo_rs::RawImage =
            priv_state.raw_image_cache.as_ref().expect("just set");
        let num_components: usize = raw.num_components;

        // Build a local snapshot of plane sizes before the mutable borrow.
        let plane_widths: Vec<usize> = raw.plane_widths.clone();
        let plane_heights: Vec<usize> = raw.plane_heights.clone();

        // Number of `v_samp_factor` values comes from the public `comp_info` array
        // populated by `jpeg_read_header`.
        let comp_info_slice: &[JpegComponentInfoPublic] =
            if c.comp_info.is_null() || c.num_components as usize != num_components {
                // Fallback: derive v_samp_factor from plane height ratios when
                // comp_info is unavailable (unusual but defensive).
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(c.comp_info, num_components) }
            };

        // For each component, copy `comp_v_samp_factor * dct_size` rows of
        // `plane_width` samples into the caller's row pointers.
        for comp_idx in 0..num_components {
            // Derive v_samp_factor for this component.
            let vsf: usize = if comp_idx < comp_info_slice.len() {
                comp_info_slice[comp_idx].v_samp_factor.max(1) as usize
            } else {
                // Fallback: use plane height proportional to luma.
                // For 4:2:0 luma=max_vsf, chroma=max_vsf/2.
                let luma_h: usize = plane_heights.first().copied().unwrap_or(1);
                let this_h: usize = plane_heights.get(comp_idx).copied().unwrap_or(1);
                // v_samp_factor ≈ max_vsf * this_h / luma_h, clamped ≥ 1.
                ((max_vsf * this_h + luma_h / 2) / luma_h).max(1)
            };
            let rows_this_call: usize = vsf * dct_size;
            let plane_width: usize = plane_widths.get(comp_idx).copied().unwrap_or(0);
            let plane_height: usize = plane_heights.get(comp_idx).copied().unwrap_or(0);
            let rows_already: usize = priv_state
                .raw_rows_consumed
                .get(comp_idx)
                .copied()
                .unwrap_or(0);

            // Get the outer pointer for this component: data[comp_idx].
            let comp_outer_ptr: *mut *mut u8 = unsafe { *data.add(comp_idx) };
            if comp_outer_ptr.is_null() {
                continue;
            }

            for row_in_imcu in 0..rows_this_call {
                let src_row: usize = rows_already + row_in_imcu;
                if src_row >= plane_height {
                    break;
                }
                let dst_row_ptr: *mut u8 = unsafe { *comp_outer_ptr.add(row_in_imcu) };
                if dst_row_ptr.is_null() {
                    continue;
                }
                let src_plane: &[u8] = &raw.planes[comp_idx];
                let src_offset: usize = src_row * plane_width;
                let src_slice: &[u8] = &src_plane[src_offset..src_offset + plane_width];
                unsafe {
                    std::ptr::copy_nonoverlapping(src_slice.as_ptr(), dst_row_ptr, plane_width);
                }
            }

            if let Some(consumed) = priv_state.raw_rows_consumed.get_mut(comp_idx) {
                *consumed += rows_this_call;
            }
        }

        let delivered: JDimension = rows_per_imcu as JDimension;
        c.output_scanline = c.output_scanline.saturating_add(delivered);
        priv_state.last_error = CString::new("No error").expect("static");
        delivered
    })
}

/// `jpeg12_read_raw_data(cinfo, data, max_lines) -> JDIMENSION`.
///
/// 12-bit counterpart to `jpeg_read_raw_data`. Delivers one iMCU row
/// of 12-bit samples per component into a JSAMPIMAGE16
/// (`*mut *mut *mut i16`) array. Lazily materialises the whole
/// stream through `libjpeg_turbo_rs::raw_data_12::decompress_raw_12`
/// on the first call and copies subsequent iMCU rows out of the
/// cached planes.
///
/// Returns 0 (and populates `priv_state.last_error`) on precondition
/// failures: wrong precision, missing data pointer, `max_lines`
/// smaller than one iMCU row, or a Rust-side decode error. Returns
/// `max_v_samp_factor * DCT_v_scaled_size` rows on success.
#[no_mangle]
pub extern "C" fn jpeg12_read_raw_data(
    cinfo: *mut c_void,
    data: *mut *mut *mut i16,
    max_lines: JDimension,
) -> JDimension {
    crate::unwind_guard!(0, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        let priv_state: &mut DecompressPrivate = match unsafe { priv_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return 0,
        };

        // 12-bit gate: only `data_precision == 12` reaches this entry
        // point; 8-bit callers should use `jpeg_read_raw_data`.
        if c.data_precision != 12 {
            priv_state.last_error = CString::new(format!(
                "jpeg12_read_raw_data: JERR_BAD_PRECISION (data_precision={} not 12)",
                c.data_precision
            ))
            .unwrap_or_default();
            return 0;
        }

        // EOF sentinel: output_scanline >= output_height → return 0
        // (matches libjpeg's JWRN_TOO_MUCH_DATA path).
        if c.output_scanline >= c.output_height {
            return 0;
        }

        if data.is_null() {
            priv_state.last_error =
                CString::new("jpeg12_read_raw_data: data pointer is NULL").unwrap_or_default();
            return 0;
        }

        let max_vsf: usize = c.max_v_samp_factor as usize;
        let dct_size: usize = if c.min_DCT_v_scaled_size > 0 {
            c.min_DCT_v_scaled_size as usize
        } else {
            DCTSIZE
        };
        let rows_per_imcu: usize = max_vsf * dct_size;

        if (max_lines as usize) < rows_per_imcu {
            priv_state.last_error = CString::new(format!(
            "jpeg12_read_raw_data: JERR_BUFFER_SIZE — max_lines ({max_lines}) < rows_per_iMCU ({rows_per_imcu})"
        )).unwrap_or_default();
            return 0;
        }

        if priv_state.raw_image_cache_12.is_none() {
            let bytes: &[u8] = match priv_state.source.as_bytes() {
                Some(b) => b,
                None => {
                    priv_state.last_error =
                        CString::new("jpeg12_read_raw_data: no source").unwrap_or_default();
                    return 0;
                }
            };
            match libjpeg_turbo_rs::raw_data_12::decompress_raw_12(bytes) {
                Ok(raw) => {
                    let ncomp: usize = raw.num_components;
                    priv_state.raw_rows_consumed = vec![0usize; ncomp];
                    priv_state.raw_image_cache_12 = Some(raw);
                }
                Err(e) => {
                    priv_state.last_error = CString::new(format!(
                        "jpeg12_read_raw_data: decompress_raw_12 failed: {e}"
                    ))
                    .unwrap_or_default();
                    return 0;
                }
            }
        }

        let raw: &libjpeg_turbo_rs::raw_data_12::RawImage12 =
            priv_state.raw_image_cache_12.as_ref().expect("just set");
        let num_components: usize = raw.num_components;

        let plane_widths: Vec<usize> = raw.plane_widths.clone();
        let plane_heights: Vec<usize> = raw.plane_heights.clone();

        let comp_info_slice: &[JpegComponentInfoPublic] =
            if c.comp_info.is_null() || c.num_components as usize != num_components {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(c.comp_info, num_components) }
            };

        for comp_idx in 0..num_components {
            let vsf: usize = if comp_idx < comp_info_slice.len() {
                comp_info_slice[comp_idx].v_samp_factor.max(1) as usize
            } else {
                let luma_h: usize = plane_heights.first().copied().unwrap_or(1);
                let this_h: usize = plane_heights.get(comp_idx).copied().unwrap_or(1);
                ((max_vsf * this_h + luma_h / 2) / luma_h).max(1)
            };
            let rows_this_call: usize = vsf * dct_size;
            let plane_width: usize = plane_widths.get(comp_idx).copied().unwrap_or(0);
            let plane_height: usize = plane_heights.get(comp_idx).copied().unwrap_or(0);
            let rows_already: usize = priv_state
                .raw_rows_consumed
                .get(comp_idx)
                .copied()
                .unwrap_or(0);

            let comp_outer_ptr: *mut *mut i16 = unsafe { *data.add(comp_idx) };
            if comp_outer_ptr.is_null() {
                continue;
            }

            for row_in_imcu in 0..rows_this_call {
                let src_row: usize = rows_already + row_in_imcu;
                if src_row >= plane_height {
                    break;
                }
                let dst_row_ptr: *mut i16 = unsafe { *comp_outer_ptr.add(row_in_imcu) };
                if dst_row_ptr.is_null() {
                    continue;
                }
                let src_plane: &[i16] = &raw.planes[comp_idx];
                let src_offset: usize = src_row * plane_width;
                let src_slice: &[i16] = &src_plane[src_offset..src_offset + plane_width];
                // SAFETY: src/dst point to `plane_width` valid i16
                // samples each; copy_nonoverlapping count is in *items*.
                unsafe {
                    std::ptr::copy_nonoverlapping(src_slice.as_ptr(), dst_row_ptr, plane_width);
                }
            }

            if let Some(consumed) = priv_state.raw_rows_consumed.get_mut(comp_idx) {
                *consumed += rows_this_call;
            }
        }

        let delivered: JDimension = rows_per_imcu as JDimension;
        c.output_scanline = c.output_scanline.saturating_add(delivered);
        priv_state.last_error = CString::new("No error").expect("static");
        delivered
    })
}

// ---------------------------------------------------------------------------
// Incremental-input marker scanning (P4-13).
//
// These pure helpers let `jpeg_consume_input` drive a multi-scan stream in
// lock-step with a suspending source manager: `find_first_sos` reports when the
// accumulated bytes contain a complete header (through the first SOS), and
// `scan_next_boundary` walks the entropy-coded body to the next scan / EOI,
// returning `NeedMore` when the buffered prefix is too short — the signal to
// pull another chunk (or report `JPEG_SUSPENDED`).
// ---------------------------------------------------------------------------

// P4-58: the boundary scanner moved to the library crate
// (`libjpeg_turbo_rs::decode::boundary`) so the Rust-native incremental
// reader and this suspension core share one mechanism (P4-26). The
// types/functions below are re-imports of the identical code.
use libjpeg_turbo_rs::decode::boundary::{find_first_sos, scan_next_boundary, MarkerBoundary};

/// Hard ceiling on the accumulated body so a source that never delivers
/// EOI cannot grow the buffer without bound (P4-13).
const P4_13_MAX_BODY_BYTES: usize = 256 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Buffered-image-mode shim (P0-3 follow-on).
//
// Stock djpeg/cjpeg/jpegtran do not exercise these in the default
// configuration, but Pillow's downstream wrappers — and any caller
// that toggles `cinfo->buffered_image = TRUE` — resolve them at link
// time. We provide thin, non-buffered stubs that match the libjpeg
// contract for "buffered image mode disabled", which is the de-facto
// default. Concretely:
//   * `jpeg_consume_input` returns `JPEG_REACHED_EOI` once the upfront
//     decoder has populated the source buffer (we always read end-to-end).
//   * `jpeg_input_complete` returns TRUE for the same reason.
//   * `jpeg_has_multiple_scans` reflects `c.progressive_mode` from the
//     header, matching upstream `jpeg_has_multiple_scans`.
//   * `jpeg_start_output` / `jpeg_finish_output` succeed for any scan
//     since we always have the full image decoded.
//   * `jpeg_new_colormap` is a no-op (we don't ship the 1-pass
//     quantizer; quantize paths run through the higher-level Rust
//     library).
// ---------------------------------------------------------------------------

// `JPEG_SUSPENDED` already defined above (it shares the SUSPENDED=0
// code with `jpeg_read_header`); the SOS/EOI codes only exist on the
// `jpeg_consume_input` return path.
const JPEG_REACHED_SOS: c_int = 1;
const JPEG_REACHED_EOI: c_int = 2;

/// `jpeg_consume_input(cinfo) -> int` — see `jpeglib.h:1108`.
///
/// Stock libjpeg drives the input state machine here: callers may loop
/// until it returns `JPEG_REACHED_EOI`, expecting that headers get
/// parsed and `cinfo->global_state` advances. Returning
/// `JPEG_REACHED_EOI` unconditionally would let buffered/progressive
/// callers skip `jpeg_read_header` entirely and then read
/// uninitialised public fields. We therefore drive the state machine:
///
///   * `DSTATE_START` → invoke `jpeg_read_header` (returns
///     `JPEG_HEADER_OK` and leaves state at `DSTATE_INHEADER` on
///     success, or stays at `DSTATE_INHEADER` + returns `JPEG_SUSPENDED`
///     on a partial source). On success we advance state past
///     `DSTATE_INHEADER` so the next call surfaces `JPEG_REACHED_EOI`
///     instead of looping.
///   * `DSTATE_INHEADER` → retry `jpeg_read_header` to resume from a
///     prior suspension; on success advance state.
///   * Anything past `DSTATE_INHEADER` (post header-parse) → return
///     `JPEG_REACHED_EOI` because our shim buffers the entire stream
///     up front.
#[no_mangle]
pub extern "C" fn jpeg_consume_input(cinfo: *mut c_void) -> c_int {
    crate::unwind_guard!(JPEG_SUSPENDED, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return JPEG_SUSPENDED,
        };
        // Short-circuit if the header has already been parsed: rerunning
        // `jpeg_read_header` would clobber the caller's post-header
        // tweaks (out_color_space, comp_info, quantize_colors, …). For our
        // fully-buffered shim, EOI is the truthful answer the moment a
        // header is in hand.
        //
        // We must also advance `global_state` to `DSTATE_SCANNING` here so
        // that `jpeg_input_complete()` (which gates on
        // `global_state >= DSTATE_SCANNING`) reports TRUE. Otherwise a
        // caller polling `while (!jpeg_input_complete()) jpeg_consume_input()`
        // — the buffered/progressive idiom — would loop forever even
        // though we keep returning `JPEG_REACHED_EOI`.
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);

        // P4-13: a suspending source delivered the header (through the first
        // SOS) but not yet the whole body. Drive the remaining body in lock-step
        // with the source manager: report the next scan boundary, or
        // `JPEG_SUSPENDED` when the source is dry, exactly as upstream
        // `consume_input` does. `c` and the private state live in disjoint
        // allocations, so holding both `&mut` here matches the established
        // pattern (`jpeg_start_decompress`).
        if let Some(priv_state) = unsafe { priv_from_ptr(priv_ptr) } {
            if priv_state.body_incomplete {
                loop {
                    let cursor: usize = priv_state.body_scan_cursor;
                    let boundary: MarkerBoundary = match priv_state.source.as_bytes() {
                        Some(b) => scan_next_boundary(b, cursor),
                        None => MarkerBoundary::NeedMore(cursor),
                    };
                    match boundary {
                        MarkerBoundary::Sos(off) => {
                            priv_state.body_scan_cursor = off;
                            // Mirror upstream: each consumed SOS advances the
                            // public scan counter (read_header set it to 1 at
                            // the first SOS), so a buffered-image consumer that
                            // reads `input_scan_number` / passes it to
                            // `jpeg_start_output` stays in lock-step.
                            c.input_scan_number = c.input_scan_number.saturating_add(1);
                            return JPEG_REACHED_SOS;
                        }
                        MarkerBoundary::Eoi(off) => {
                            priv_state.body_scan_cursor = off;
                            priv_state.eoi_seen = true;
                            priv_state.body_incomplete = false;
                            if c.global_state < DSTATE_SCANNING {
                                c.global_state = DSTATE_SCANNING;
                            }
                            // Body complete: refresh marker_list so a caller that
                            // only drains via consume_input still sees inter-scan
                            // APP/COM markers (P4-13).
                            rebuild_marker_list_from_source(c, priv_state);
                            return JPEG_REACHED_EOI;
                        }
                        MarkerBoundary::NeedMore(resume) => {
                            // Resume the next scan from where this one stalled so
                            // a small-packet source doesn't rescan O(n²).
                            priv_state.body_scan_cursor = resume;
                            let pull: PullResult = match priv_state.source.as_owned_mut() {
                                // SAFETY: `c` and `priv_state` are disjoint
                                // allocations; the fill callback touches only the
                                // caller's source manager, never `buf`.
                                Some(buf) => unsafe { pull_more_from_source_mgr(c, buf) },
                                None => return JPEG_SUSPENDED,
                            };
                            match pull {
                                PullResult::Progressed => {
                                    if priv_state.source.as_bytes().map_or(0, <[u8]>::len)
                                        > P4_13_MAX_BODY_BYTES
                                    {
                                        // Cap tripped: clear `body_incomplete` so a
                                        // retry can't re-enter this loop and append
                                        // past the cap (bounds memory; terminal).
                                        priv_state.body_incomplete = false;
                                        priv_state.last_error = CString::new(
                                            "jpeg_consume_input: suspended body exceeds 256 MiB cap",
                                        )
                                        .unwrap_or_default();
                                        return JPEG_SUSPENDED;
                                    }
                                    continue;
                                }
                                PullResult::Suspended => return JPEG_SUSPENDED,
                                PullResult::Ended => {
                                    // Stream ended without an EOI (truncated).
                                    // Mark the body complete so the buffered
                                    // decode can run best-effort, matching the
                                    // lenient drain that `drain_caller_source_mgr`
                                    // applies to a callback-less source.
                                    priv_state.eoi_seen = true;
                                    priv_state.body_incomplete = false;
                                    if c.global_state < DSTATE_SCANNING {
                                        c.global_state = DSTATE_SCANNING;
                                    }
                                    rebuild_marker_list_from_source(c, priv_state);
                                    return JPEG_REACHED_EOI;
                                }
                            }
                        }
                    }
                }
            }
        }

        let header_done: bool = match unsafe { priv_from_ptr(priv_ptr) } {
            Some(p) => p.header_parsed_ok,
            None => false,
        };
        if header_done {
            if c.global_state < DSTATE_SCANNING {
                c.global_state = DSTATE_SCANNING;
            }
            return JPEG_REACHED_EOI;
        }
        match c.global_state {
            DSTATE_START | DSTATE_INHEADER => {
                // `require_image=FALSE`: stock libjpeg's `jpeg_consume_input`
                // accepts a tables-only abbreviated datastream as a valid
                // input — it returns `JPEG_REACHED_EOI` rather than raising
                // `JERR_NO_IMAGE` (which is the public `jpeg_read_header(…,
                // require_image=TRUE)` contract). Passing `0` here lets the
                // tables-only branch return `JPEG_HEADER_TABLES_ONLY`, which
                // the match arm below maps to `JPEG_REACHED_EOI`.
                let result: c_int = jpeg_read_header(cinfo, /*require_image=*/ 0);
                // Re-read the state because `jpeg_read_header` mutated it.
                let c2: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
                    Some(c) => c,
                    None => return JPEG_SUSPENDED,
                };
                match result {
                    JPEG_HEADER_OK => {
                        // Our `jpeg_read_header` leaves state at
                        // `DSTATE_INHEADER` on success (upstream uses
                        // `DSTATE_READY`, which we don't model
                        // separately). Advance past the header-parse
                        // arm so subsequent `consume_input` polls report
                        // `JPEG_REACHED_EOI` instead of looping forever
                        // on `JPEG_REACHED_SOS`. `jpeg_start_decompress`
                        // re-asserts `DSTATE_SCANNING` regardless, so
                        // skipping ahead is safe.
                        c2.global_state = DSTATE_SCANNING;
                        JPEG_REACHED_SOS
                    }
                    JPEG_HEADER_TABLES_ONLY => {
                        // Mirror stock libjpeg's `inputctl->eoi_reached = TRUE`
                        // for tables-only inputs by advancing past
                        // `DSTATE_SCANNING` so `jpeg_input_complete` returns
                        // `TRUE`. Without this, a documented buffered-image
                        // polling loop —
                        //
                        //     while (!jpeg_input_complete(&cinfo))
                        //         (void) jpeg_consume_input(&cinfo);
                        //
                        // — never terminates on a tables-only datastream
                        // because the EOI return below isn't observed by
                        // `jpeg_input_complete`'s `global_state >=
                        // DSTATE_SCANNING` gate. `jpeg_start_decompress`
                        // re-asserts `DSTATE_SCANNING` regardless, so
                        // skipping ahead does not stomp on a real image
                        // header that arrives later (it can't — we just
                        // returned EOI).
                        c2.global_state = DSTATE_SCANNING;
                        JPEG_REACHED_EOI
                    }
                    _ => JPEG_SUSPENDED,
                }
            }
            _ => JPEG_REACHED_EOI,
        }
    })
}

/// `jpeg_input_complete(cinfo) -> boolean` — see `jpeglib.h:1106`.
///
/// TRUE only once the header has been parsed (state ≥ `DSTATE_SCANNING`).
/// Earlier states would mislead a caller polling for "input ready"
/// after just installing a source.
#[no_mangle]
pub extern "C" fn jpeg_input_complete(cinfo: *mut c_void) -> CBoolean {
    crate::unwind_guard!(0, {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        if c.global_state < DSTATE_SCANNING {
            return 0;
        }
        // P4-13: while the body is still draining from a suspending source the
        // input is NOT complete, even though the header is parsed and the state
        // has advanced — so the buffered-image polling idiom
        // (`while (!jpeg_input_complete()) jpeg_consume_input();`) keeps driving
        // until `jpeg_consume_input` reaches EOI and clears `body_incomplete`.
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        if let Some(p) = unsafe { priv_from_ptr(priv_ptr) } {
            if p.body_incomplete {
                return 0;
            }
        }
        1
    })
}

/// `jpeg_has_multiple_scans(cinfo) -> boolean`.
///
/// Returns the `progressive_mode` flag populated by `jpeg_read_header`.
/// This is what upstream `jpeg_has_multiple_scans` does — see
/// `references/libjpeg-turbo/src/jdmaster.c::jpeg_has_multiple_scans`.
#[no_mangle]
pub extern "C" fn jpeg_has_multiple_scans(cinfo: *mut c_void) -> CBoolean {
    crate::unwind_guard!(0, {
        match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c.progressive_mode,
            None => 0,
        }
    })
}

/// `jpeg_start_output(cinfo, scan_number) -> boolean` — buffered-image
/// multi-pass output entry. We always hold the fully decoded image, so
/// any scan number succeeds. Records `scan_number` into
/// `cinfo->output_scan_number` as upstream does, so the documented
/// buffered-image loop (which stops once
/// `jpeg_input_complete() && input_scan_number == output_scan_number`)
/// terminates for the P4-13 suspending path where `input_scan_number`
/// advances per scan.
#[no_mangle]
pub extern "C" fn jpeg_start_output(cinfo: *mut c_void, scan_number: c_int) -> CBoolean {
    crate::unwind_guard!(0, {
        if let Some(c) = unsafe { cinfo_mut(cinfo) } {
            c.output_scan_number = scan_number;
        }
        1
    })
}

/// `jpeg_finish_output(cinfo) -> boolean` — buffered-image multi-pass
/// finish. No-op success in our non-buffered model.
#[no_mangle]
pub extern "C" fn jpeg_finish_output(_cinfo: *mut c_void) -> CBoolean {
    crate::unwind_guard!(0, { 1 })
}

/// `jpeg_new_colormap(cinfo)` — buffered-image colormap update. We
/// don't ship the upstream 1-pass color quantizer, so this is a no-op.
#[no_mangle]
pub extern "C" fn jpeg_new_colormap(_cinfo: *mut c_void) {
    crate::unwind_guard!((), {})
}

// ---------------------------------------------------------------------------
// Abort / generic destroy entry points (P0-3 follow-on).
//
// These names are part of the documented public ABI; downstream
// callers resolve them through `dlsym` for their teardown paths.
// `jpeg_abort_*` reset state without freeing the cinfo allocation;
// `jpeg_destroy` is the polymorphic wrapper that dispatches based on
// `cinfo->is_decompressor`.
// ---------------------------------------------------------------------------

/// `jpeg_abort_compress(cinfo)` — reset compress state for reuse.
///
/// Upstream `jpeg_abort_compress` returns the compressor to
/// `CSTATE_START`, drops any per-pass scratch state, and clears
/// pending coefficient handles so the same `cinfo` can be reused for
/// a new image. Without this, error-recovery flows that abort a
/// partial encode and re-issue `jpeg_start_compress` would observe
/// stale `next_scanline` / pixel buffers from the failed run, and a
/// reuse without `jpeg_write_icc_profile` would still emit the
/// aborted run's APP2 ICC chunks.
#[no_mangle]
pub extern "C" fn jpeg_abort_compress(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        let priv_ptr: *mut c_void = c.master;
        if let Some(p) = unsafe { priv_compress_from_ptr(priv_ptr) } {
            p.have_started = false;
            p.tables_only = false;
            p.pixels_u8.clear();
            p.pixels_u16.clear();
            p.pending_markers.clear();
            // Drop the pending ICC profile too — otherwise a reuse path
            // that re-runs encode without `jpeg_write_icc_profile` would
            // still inject the aborted image's APP2 ICC chunks.
            p.icc_profile = None;
            p.pending_coef_arrays = std::ptr::null();
            // Upstream keeps the destination manager across abort so a second
            // image can be written without re-calling `jpeg_stdio_dest` /
            // `jpeg_mem_dest` (jdatadst.c:196-198). Keep it — but clear any
            // failure it recorded, or the aborted image's I/O error would be
            // reported against the next one.
            if let Some(ref mut mgr) = p.dest_mgr {
                mgr.pending_error = None;
            }
        }
        c.global_state = CSTATE_START;
        c.next_scanline = 0;
    })
}

/// `jpeg_abort_decompress(cinfo)` — reset decompress state for reuse.
///
/// Drops the cached decode + coefficient arrays, clears any source
/// bridged from a caller-installed `cinfo->src`, and returns
/// `global_state` to `DSTATE_START` so a follow-up
/// `jpeg_read_header` re-runs the source bridge against whatever
/// new source the caller has installed.
#[no_mangle]
pub extern "C" fn jpeg_abort_decompress(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        let c: &mut JpegDecompressPublic = match unsafe { cinfo_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        let priv_ptr: *mut c_void = decompress_private_raw(cinfo);
        if let Some(p) = unsafe { priv_from_ptr(priv_ptr) } {
            p.decoded = None;
            // Unhook the foreign array from the global side table BEFORE
            // dropping the `CoefHandle` it was registered against; if we
            // dropped the handle first, a follow-up
            // `jpeg_write_coefficients(other_dst, our_array_ptr)` could
            // hit the side table and dereference freed memory.
            //
            // Then release the JPOOL_IMAGE allocations that backed the
            // foreign barray array. See the matching comment in
            // `jpeg_finish_decompress` for the rationale (codex round-8
            // review of b7f690d — without this the abort path leaks the
            // previous image's virtual arrays).
            if !p.coef_array_ptr.is_null() {
                coef_unregister_array(p.coef_array_ptr as *const c_void);
                p.coef_array_ptr = std::ptr::null_mut();
                release_decompress_image_pool(cinfo, c.mem);
            }
            p.coefficients = None;
            p.crop_active = false;
            p.crop_xoffset = 0;
            p.crop_width = 0;
            // Drop any source bytes bridged from the previous run so
            // `jpeg_read_header` re-bridges from the (possibly new)
            // `cinfo->src` next time. Also clear the bridge resume buffer
            // so a non-blocking source from a future image doesn't
            // accidentally resume in the middle of an unrelated stream.
            p.source = JpegSource::None;
            p.bridge_partial.clear();
            // Force a re-parse of the next image's header.
            p.header_parsed_ok = false;
            // P4-13 incremental-input state, reset on abort.
            p.body_incomplete = false;
            p.body_scan_cursor = 0;
            p.eoi_seen = false;
            c.input_scan_number = 0;
            // Drop the previous image's saved-marker linked list. Stock
            // libjpeg-turbo's `jpeg_abort` releases `JPOOL_IMAGE` (which
            // backs `marker_list`) and the next `jpeg_read_header` starts
            // with a clean slate. We mirror that here so a follow-up
            // `jpeg_read_header` that fails early (returns
            // `JPEG_SUSPENDED` before reaching the marker builder) cannot
            // leave `cinfo->marker_list` pointing at the previous image's
            // markers — caught by the post-implementation review.
            c.marker_list = std::ptr::null_mut();
            p.marker_list_storage.clear();
            // Drop the lazy raw-data caches; reuse on a new JPEG must
            // re-materialise. Same rationale as the matching block in
            // `jpeg_finish_decompress`.
            p.raw_image_cache = None;
            p.raw_image_cache_12 = None;
            p.raw_rows_consumed.clear();
        }
        c.global_state = DSTATE_START;
        c.output_scanline = 0;
    })
}

/// `jpeg_abort(cinfo)` — common-struct abort. Dispatches via the
/// `is_decompressor` flag at offset 32 of the common prefix.
#[no_mangle]
pub extern "C" fn jpeg_abort(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        // Both struct prefixes carry `is_decompressor` at offset 32. We
        // only need to read the flag through whichever struct was
        // originally allocated; field offset is identical.
        if cinfo.is_null() {
            return;
        }
        let is_decompressor: CBoolean = unsafe { *(cinfo as *const u8).add(32).cast::<CBoolean>() };
        if is_decompressor != 0 {
            jpeg_abort_decompress(cinfo);
        } else {
            jpeg_abort_compress(cinfo);
        }
    })
}

/// `jpeg_destroy(cinfo)` — common-struct destroy. Same dispatch as
/// `jpeg_abort`.
#[no_mangle]
pub extern "C" fn jpeg_destroy(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        if cinfo.is_null() {
            return;
        }
        let is_decompressor: CBoolean = unsafe { *(cinfo as *const u8).add(32).cast::<CBoolean>() };
        if is_decompressor != 0 {
            jpeg_destroy_decompress(cinfo);
        } else {
            jpeg_destroy_compress(cinfo);
        }
    })
}

// ---------------------------------------------------------------------------
// Allocation helpers (P0-3 follow-on / P0-4 prep).
//
// Stock `transupp.c` and any caller that builds quant/huff tables on
// the fly resolves these. They allocate a zero-initialised
// `JQUANT_TBL` / `JHUFF_TBL` from the cinfo's memory manager. We
// allocate via libc malloc tagged with the same SAFETY contract
// memmgr uses elsewhere — caller must release via the same memory
// manager (`jpeg_destroy_*` will clean up if registered).
// ---------------------------------------------------------------------------

/// Allocate `size` zero-initialised bytes through `cinfo->mem->alloc_small`
/// in the permanent pool, so `jpeg_destroy_*` can free the storage by
/// releasing the pool. Falls back to the global allocator only when
/// `cinfo` is NULL or the memory manager isn't installed (pre-create
/// callers); the leak there is documented.
///
/// Both `j_compress_ptr` and `j_decompress_ptr` carry `mem` at offset
/// `1 * sizeof(*void)` of the common prefix, so this function works
/// for either side.
unsafe fn alloc_through_memmgr_or_heap(cinfo: *mut c_void, size: usize) -> *mut u8 {
    if !cinfo.is_null() {
        // SAFETY: caller asserts `cinfo` points to a libjpeg cinfo
        // whose common prefix matches our `JpegDecompressPublic` /
        // `JpegCompressPublic` mirrors. `mem` lives at offset
        // `1 * sizeof(*void)` (immediately after `err`).
        let mem_field: *const *mut c_void =
            unsafe { (cinfo as *const u8).add(8).cast::<*mut c_void>() };
        let mem_ptr: *mut c_void = unsafe { *mem_field };
        if !mem_ptr.is_null() {
            // SAFETY: mem_ptr is a `JpegMemoryMgr` per our
            // create_memory_mgr contract.
            let mem: &memmgr::JpegMemoryMgr =
                unsafe { &*(mem_ptr as *const memmgr::JpegMemoryMgr) };
            if let Some(alloc_small) = mem.alloc_small {
                // SAFETY: caller-provided alloc_small honoring the
                // libjpeg `void *(j_common_ptr, int, size_t)` proto.
                let raw: *mut c_void = unsafe { alloc_small(cinfo, memmgr::JPOOL_PERMANENT, size) };
                if !raw.is_null() {
                    // libjpeg's alloc_small does not zero-fill; do it here.
                    unsafe {
                        std::ptr::write_bytes(raw as *mut u8, 0, size);
                    }
                    return raw as *mut u8;
                }
            }
        }
    }
    // Fallback: global allocator (the destroy path won't free this,
    // which is acceptable for the rare pre-create / no-mem case
    // libjpeg itself does not formally support).
    let layout: std::alloc::Layout =
        std::alloc::Layout::from_size_align(size, std::mem::align_of::<u64>())
            .unwrap_or_else(|_| std::alloc::Layout::from_size_align(size, 8).unwrap());
    // SAFETY: layout has size>0 (struct sizes used here are >0).
    unsafe { std::alloc::alloc_zeroed(layout) }
}

/// `jpeg_alloc_quant_table(cinfo) -> JQUANT_TBL*`. The returned table
/// is zero-initialised per upstream contract (`sent_table = FALSE`).
/// Allocated through `cinfo->mem->alloc_small(JPOOL_PERMANENT, …)` so
/// the table is released by `jpeg_destroy_*` along with the rest of
/// the permanent pool; this matches upstream `jpeg_alloc_quant_table`
/// at `references/libjpeg-turbo/src/jcomapi.c`.
#[no_mangle]
pub extern "C" fn jpeg_alloc_quant_table(cinfo: *mut c_void) -> *mut JQuantTblPublic {
    crate::unwind_guard!(std::ptr::null_mut(), {
        // SAFETY: passing through to the memory manager owned by the
        // caller's `cinfo`.
        let raw: *mut u8 =
            unsafe { alloc_through_memmgr_or_heap(cinfo, std::mem::size_of::<JQuantTblPublic>()) };
        raw.cast::<JQuantTblPublic>()
    })
}

/// `jpeg_alloc_huff_table(cinfo) -> JHUFF_TBL*`. Zero-initialised.
/// Allocated via the memory manager — see `jpeg_alloc_quant_table`.
#[no_mangle]
pub extern "C" fn jpeg_alloc_huff_table(cinfo: *mut c_void) -> *mut JHuffTblPublic {
    crate::unwind_guard!(std::ptr::null_mut(), {
        // SAFETY: passing through to the memory manager owned by the
        // caller's `cinfo`.
        let raw: *mut u8 =
            unsafe { alloc_through_memmgr_or_heap(cinfo, std::mem::size_of::<JHuffTblPublic>()) };
        raw.cast::<JHuffTblPublic>()
    })
}

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

/// libjpeg's `OUTPUT_BUF_SIZE` (jdatadst.c:38). It is the stdio staging size
/// *and* the initial allocation `jpeg_mem_dest` makes when the caller supplies
/// no buffer, so the value is observable through `*outsize` and must match.
const OUTPUT_BUF_SIZE: usize = 4096;

/// Mirror of upstream `my_mem_destination_mgr` (jdatadst.c:43-51).
///
/// The ownership split is the whole point: `buffer` is whatever we are
/// currently filling — possibly memory the *application* owns — while
/// `newbuffer` is the only block this library ever allocated and therefore the
/// only one it may free. `bufsize` is a capacity, never a payload length.
struct MemDestState {
    outbuffer: *mut *mut u8,
    outsize: *mut std::os::raw::c_ulong,
    /// Library-allocated block, or null while the caller's buffer is in use.
    newbuffer: *mut u8,
    /// Start of the buffer currently being filled.
    buffer: *mut u8,
    /// Capacity of `buffer` in bytes.
    bufsize: usize,
}

/// Mirror of upstream `my_destination_mgr` (jdatadst.c:29-34).
struct StdioDestState {
    outfile: *mut c_void,
}

/// Destination variants, each carrying the private state its upstream
/// counterpart keeps behind the public `jpeg_destination_mgr`.
///
/// There is no "no destination" variant: that state is `dest_mgr: None`, since
/// a manager only ever exists because `jpeg_stdio_dest` or `jpeg_mem_dest`
/// created it with a concrete kind.
enum JpegDest {
    /// Application-owned `unsigned char **outbuffer` + `unsigned long *outsize`.
    Mem(MemDestState),
    /// `FILE *` written through `fwrite`, flushed by `term_destination`.
    File(StdioDestState),
}

/// Private encoder state held behind `JpegCompressPublic::priv_ptr`.
#[allow(dead_code)] // fields populated across C2-1..C2-5 subtasks
struct CompressPrivate {
    /// The library-provided destination manager, if one is installed, together
    /// with the private state its callbacks need. Deliberately *not* split
    /// into separate `CompressPrivate` fields — see `OwnedDestMgr`.
    dest_mgr: Option<Box<OwnedDestMgr>>,
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
    /// natural order, matching upstream `JQUANT_TBL::quantval` and the native
    /// encoder input. Unused slots are `None`.
    quant_tables: Vec<Option<[u16; 64]>>,
    /// Coefficient handle stashed by `jpeg_write_coefficients`. The
    /// pointer is interpreted as `*const libjpeg_turbo_rs::JpegCoefficients`
    /// at finish time. Owned by the source `j_decompress_ptr`'s private
    /// state (see `jpeg_read_coefficients`); caller must keep that cinfo
    /// alive across the matching `jpeg_finish_compress` call.
    pending_coef_arrays: *const c_void,
    /// Per-component plane buffers accumulating caller-supplied raw
    /// (pre-downsampled) rows. Each `Vec<u8>` is sized
    /// `MCU-aligned_width × MCU-aligned_height` bytes for the component.
    /// Populated by `jpeg_write_raw_data` calls; consumed by
    /// `jpeg_finish_compress` via `libjpeg_turbo_rs::compress_raw`.
    raw_plane_buffers: Vec<Vec<u8>>,
    /// Number of rows already written into `raw_plane_buffers[i]` /
    /// `raw_plane_buffers_12[i]`. Reused across both 8-bit and 12-bit
    /// raw-data accumulators since the row-count semantics are
    /// identical (it counts logical rows, not bytes).
    raw_rows_filled: Vec<usize>,
    /// MCU-aligned width of each raw plane (samples per row).
    raw_plane_widths: Vec<usize>,
    /// MCU-aligned height of each raw plane (total rows in buffer).
    raw_plane_heights: Vec<usize>,
    /// 12-bit counterpart to `raw_plane_buffers`. Populated by
    /// `jpeg12_write_raw_data` (each i16 holds a 12-bit sample
    /// 0..4095); consumed by `jpeg_finish_compress` when
    /// `data_precision == 12` via `libjpeg_turbo_rs::compress_raw_12`.
    /// Empty when the 8-bit path is in use; non-empty otherwise.
    /// `raw_rows_filled` / `raw_plane_widths` / `raw_plane_heights` are
    /// shared with the 8-bit path — the 8-bit and 12-bit accumulators
    /// are mutually exclusive on a single cinfo (selected by
    /// `data_precision`).
    raw_plane_buffers_12: Vec<Vec<i16>>,
}

impl Default for CompressPrivate {
    fn default() -> Self {
        Self {
            dest_mgr: None,
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
            raw_plane_buffers: Vec::new(),
            raw_rows_filled: Vec::new(),
            raw_plane_widths: Vec::new(),
            raw_plane_heights: Vec::new(),
            raw_plane_buffers_12: Vec::new(),
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
    // J_COLOR_SPACE values from libjpeg-turbo's `jmorecfg.h`. The
    // extended (`JCS_EXT_*`) family starts at 6 — Pillow's
    // `_imaging.so` resolves these names by enum, so honouring 6..15
    // is required for any classic-API caller built against
    // libjpeg-turbo. (An older mapping pinned them to 13..22 which
    // broke RGB encode for Pillow — see commit log.)
    match cs {
        JCS_GRAYSCALE => Some(PixelFormat::Grayscale),
        JCS_RGB => Some(PixelFormat::Rgb),
        JCS_YCBCR => Some(PixelFormat::Rgb), // treated as RGB during encode
        JCS_CMYK => Some(PixelFormat::Cmyk),
        // Extended color spaces (libjpeg-turbo only).
        6 /* JCS_EXT_RGB */ => Some(PixelFormat::Rgb),
        7 /* JCS_EXT_RGBX */ => Some(PixelFormat::Rgbx),
        8 /* JCS_EXT_BGR */ => Some(PixelFormat::Bgr),
        9 /* JCS_EXT_BGRX */ => Some(PixelFormat::Bgrx),
        10 /* JCS_EXT_XBGR */ => Some(PixelFormat::Xbgr),
        11 /* JCS_EXT_XRGB */ => Some(PixelFormat::Xrgb),
        12 /* JCS_EXT_RGBA */ => Some(PixelFormat::Rgba),
        13 /* JCS_EXT_BGRA */ => Some(PixelFormat::Bgra),
        14 /* JCS_EXT_ABGR */ => Some(PixelFormat::Abgr),
        15 /* JCS_EXT_ARGB */ => Some(PixelFormat::Argb),
        _ => None,
    }
}

fn default_num_components_for(cs: c_int) -> c_int {
    // Mirrors `jcs_to_pixel_format_for_input`; when the JCS_EXT
    // numbering moved to its canonical 6..15 range, this table moved
    // too.
    match cs {
        JCS_GRAYSCALE => 1,
        JCS_RGB | JCS_YCBCR => 3,
        JCS_CMYK | JCS_YCCK => 4,
        6 | 8 => 3,                               // JCS_EXT_RGB / JCS_EXT_BGR
        7 | 9 | 10 | 11 | 12 | 13 | 14 | 15 => 4, // _RGBX/BGRX/XBGR/XRGB/RGBA/BGRA/ABGR/ARGB
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
    crate::unwind_guard!((), {
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
    })
}

/// Expansion of the `jpeg_create_compress(cinfo)` convenience macro.
/// libjpeg emits a direct call to `jpeg_CreateCompress` — we provide
/// both names so callers compiled against either form link cleanly.
#[no_mangle]
pub extern "C" fn jpeg_create_compress(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        jpeg_CreateCompress(cinfo, 80, std::mem::size_of::<JpegCompressPublic>());
    })
}

fn calc_compress_jpeg_dimensions(c: &mut JpegCompressPublic, lossless: bool) {
    c.jpeg_width = c.image_width;
    c.jpeg_height = c.image_height;
    let data_unit: c_int = if lossless { 1 } else { DCTSIZE as c_int };
    c.min_DCT_h_scaled_size = data_unit;
    c.min_DCT_v_scaled_size = data_unit;
}

/// `jpeg_calc_jpeg_dimensions(cinfo)`.
///
/// Mirrors libjpeg-turbo's no-compression-scaling build: JPEG image
/// dimensions are hardwired to the input image dimensions, and the scaled
/// DCT unit is 8 for lossy JPEG or 1 for lossless JPEG.
#[no_mangle]
pub extern "C" fn jpeg_calc_jpeg_dimensions(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        let lossless: bool = match unsafe { priv_compress_from_ptr(c.master) } {
            Some(p) => p.lossless_predictor != 0,
            None => false,
        };
        calc_compress_jpeg_dimensions(c, lossless);
    })
}

/// `jpeg_destroy_compress(cinfo)`.
#[no_mangle]
pub extern "C" fn jpeg_destroy_compress(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
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
    })
}

// --- destination manager callbacks -------------------------------------
//
// Two independent callback sets, exactly as upstream jdatadst.c has them, and
// — more importantly — the private state each one needs lives *inside the
// manager*, reachable only through `cinfo->dest`. Upstream does the same:
// `my_mem_destination_mgr` embeds the public `jpeg_destination_mgr` as its
// first member and every callback recovers itself with
// `(my_mem_dest_ptr)cinfo->dest` (jdatadst.c:43-53, 125).
//
// Keeping it there rather than in `CompressPrivate` is a soundness
// requirement, not a stylistic echo. The callbacks run underneath a Rust frame
// that already holds `&mut CompressPrivate`; if a callback re-derived a second
// `&mut CompressPrivate` from `cinfo->master` it would be a sibling of that
// live reference, not a reborrow of it, invalidating the caller's tag and
// making every post-flush read of this state undefined behaviour that LLVM's
// `noalias` is entitled to fold away.

/// Our destination manager plus the private state its callbacks need.
///
/// `pub_mgr` **must** stay the first field: `cinfo->dest` points at it and the
/// callbacks cast straight back, mirroring upstream's `struct
/// jpeg_destination_mgr pub` first member (jdatadst.c:30, 44).
#[repr(C)]
struct OwnedDestMgr {
    pub_mgr: JpegDestinationMgr,
    kind: JpegDest,
    /// stdio staging block; upstream's `JPOOL_IMAGE` `OUTPUT_BUF_SIZE` buffer
    /// (jdatadst.c:67-69). The memory destination never uses it — like
    /// upstream it fills the caller's buffer directly.
    staging: Vec<u8>,
    /// A failure detected inside a callback, as an upstream `JERR_*` code.
    ///
    /// Upstream raises these with `ERREXIT` from inside the callback, which
    /// `longjmp`s straight out. We cannot: the callbacks run underneath a Rust
    /// frame that still owns the encoded `Vec`, and a `longjmp` past it would
    /// leak the allocation with no owner left to free it. So the callback
    /// records the code, byte emission stops, and the public entry point
    /// raises it through `error_exit` once its own allocations are dropped.
    /// The C-visible behaviour is identical — `jpeg_finish_compress` still
    /// leaves through `error_exit` with the same `msg_code`.
    pending_error: Option<c_int>,
}

impl OwnedDestMgr {
    fn new(
        kind: JpegDest,
        init_destination: unsafe extern "C" fn(*mut c_void),
        empty_output_buffer: unsafe extern "C" fn(*mut c_void) -> CBoolean,
        term_destination: unsafe extern "C" fn(*mut c_void),
    ) -> Self {
        Self {
            pub_mgr: JpegDestinationMgr {
                next_output_byte: std::ptr::null_mut(),
                free_in_buffer: 0,
                init_destination: Some(init_destination),
                empty_output_buffer: Some(empty_output_buffer),
                term_destination: Some(term_destination),
            },
            kind,
            staging: Vec::new(),
            pending_error: None,
        }
    }
}

/// Recover our `OwnedDestMgr` from `cinfo` inside one of our own callbacks.
///
/// # Safety
///
/// Only valid from a callback this module installed. Reaching such a callback
/// means `cinfo->dest` is the `pub_mgr` field of an `OwnedDestMgr` we own — a
/// caller cannot invoke it through a foreign manager without having first
/// copied our function pointers into it, which no libjpeg contract permits.
unsafe fn owned_dest_mut<'a>(cinfo: *mut c_void) -> Option<&'a mut OwnedDestMgr> {
    let c: &mut JpegCompressPublic = unsafe { cinfo_compress_mut(cinfo) }?;
    let dest: *mut JpegDestinationMgr = c.dest;
    if dest.is_null() {
        return None;
    }
    // SAFETY: `pub_mgr` is the first field of a `#[repr(C)]` struct, so the
    // manager pointer and the owning struct share an address.
    Some(unsafe { &mut *(dest as *mut OwnedDestMgr) })
}

/// Upstream `init_mem_destination` (jdatadst.c:75-79): deliberately empty.
/// `jpeg_mem_dest` has already published `next_output_byte` / `free_in_buffer`,
/// and re-initialising here would discard bytes on a buffered-image restart.
unsafe extern "C" fn mem_init_destination(_cinfo: *mut c_void) {}

/// Upstream `empty_mem_output_buffer` (jdatadst.c:120-147): the current buffer
/// is full, so allocate a double-sized replacement, copy everything written so
/// far into it, and free *only* a block we allocated ourselves.
unsafe extern "C" fn mem_empty_output_buffer(cinfo: *mut c_void) -> CBoolean {
    let owned: &mut OwnedDestMgr = match unsafe { owned_dest_mut(cinfo) } {
        Some(o) => o,
        None => return 0,
    };
    let state: &mut MemDestState = match owned.kind {
        JpegDest::Mem(ref mut s) => s,
        // Unreachable through libjpeg's own flow: this callback is only
        // installed by `jpeg_mem_dest`, which always sets a `Mem` kind. Report
        // it as a buffer problem rather than letting it read as suspension.
        JpegDest::File(_) => {
            owned.pending_error = Some(JERR_BUFFER_SIZE);
            return 0;
        }
    };
    if state.buffer.is_null() {
        owned.pending_error = Some(JERR_BUFFER_SIZE);
        return 0;
    }

    // Exact doubling, as upstream (jdatadst.c:128). `bufsize` is never 0 here:
    // `jpeg_mem_dest` replaces a zero-capacity caller buffer with its own
    // `OUTPUT_BUF_SIZE` allocation, so this cannot degenerate to `malloc(0)`.
    let old_bufsize: usize = state.bufsize;
    let nextsize: usize = old_bufsize.saturating_mul(2);
    let nextbuffer: *mut u8 = crate::alloc::libc_malloc(nextsize);
    if nextbuffer.is_null() {
        // Upstream: ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 10).
        owned.pending_error = Some(JERR_OUT_OF_MEMORY);
        return 0;
    }
    // SAFETY: `state.buffer` is a live `old_bufsize`-byte block (either the
    // caller's buffer, whose capacity they declared, or our own allocation),
    // and `nextbuffer` is a fresh block strictly larger than `old_bufsize`.
    unsafe {
        std::ptr::copy_nonoverlapping(state.buffer, nextbuffer, old_bufsize);
    }
    // Only ever free memory this library allocated. A caller-supplied buffer
    // stays caller-owned (jdatadst.c:231-233) — freeing it would be a
    // free-of-stack/static or a double free once the caller frees it too.
    if !state.newbuffer.is_null() {
        crate::alloc::libc_free(state.newbuffer);
    }
    state.newbuffer = nextbuffer;
    state.buffer = nextbuffer;
    state.bufsize = nextsize;
    // SAFETY: `nextbuffer` has `nextsize == 2 * old_bufsize` bytes, so the
    // offset is in bounds and `old_bufsize` bytes remain writable after it.
    owned.pub_mgr.next_output_byte = unsafe { nextbuffer.add(old_bufsize) };
    owned.pub_mgr.free_in_buffer = old_bufsize;
    1
}

/// Upstream `term_mem_destination` (jdatadst.c:176-183): publish the buffer
/// actually in use and the number of bytes written into it.
unsafe extern "C" fn mem_term_destination(cinfo: *mut c_void) {
    let owned: &mut OwnedDestMgr = match unsafe { owned_dest_mut(cinfo) } {
        Some(o) => o,
        None => return,
    };
    // Upstream subtracts in `size_t` and would wrap; we clamp instead. Only an
    // application that corrupted `free_in_buffer` past `bufsize` can tell the
    // difference, and reporting 0 beats reporting a near-`SIZE_MAX` length.
    let free_in_buffer: usize = owned.pub_mgr.free_in_buffer;
    if let JpegDest::Mem(ref state) = owned.kind {
        // SAFETY: both out-pointers were validated non-NULL in `jpeg_mem_dest`
        // and belong to the caller for the lifetime of this compress handle.
        unsafe {
            *state.outbuffer = state.buffer;
            *state.outsize = state.bufsize.saturating_sub(free_in_buffer) as std::os::raw::c_ulong;
        }
    }
}

/// Upstream `init_destination` (jdatadst.c:61-73): hand the compressor an
/// `OUTPUT_BUF_SIZE` staging block to fill.
unsafe extern "C" fn stdio_init_destination(cinfo: *mut c_void) {
    let owned: &mut OwnedDestMgr = match unsafe { owned_dest_mut(cinfo) } {
        Some(o) => o,
        None => return,
    };
    owned.staging.clear();
    owned.staging.resize(OUTPUT_BUF_SIZE, 0);
    owned.pub_mgr.next_output_byte = owned.staging.as_mut_ptr();
    owned.pub_mgr.free_in_buffer = OUTPUT_BUF_SIZE;
}

/// Upstream `empty_output_buffer` (jdatadst.c:105-118): write the staging
/// block out and reset it. A short write is `JERR_FILE_WRITE`.
unsafe extern "C" fn stdio_empty_output_buffer(cinfo: *mut c_void) -> CBoolean {
    let owned: &mut OwnedDestMgr = match unsafe { owned_dest_mut(cinfo) } {
        Some(o) => o,
        None => return 0,
    };
    let file: *mut c_void = match owned.kind {
        JpegDest::File(ref s) => s.outfile,
        // See `mem_empty_output_buffer`: not reachable through libjpeg's flow.
        JpegDest::Mem(_) => {
            owned.pending_error = Some(JERR_BUFFER_SIZE);
            return 0;
        }
    };
    if file.is_null() {
        owned.pending_error = Some(JERR_FILE_WRITE);
        return 0;
    }
    // Upstream writes a hard-coded OUTPUT_BUF_SIZE because it only ever calls
    // this callback with a full buffer. Deriving the count from
    // `free_in_buffer` is identical in that case and avoids emitting
    // uninitialised staging bytes if anything ever calls it early.
    let datacount: usize = owned
        .staging
        .len()
        .saturating_sub(owned.pub_mgr.free_in_buffer);
    if datacount > 0 && write_c_file(file, &owned.staging[..datacount]) != datacount {
        owned.pending_error = Some(JERR_FILE_WRITE);
        return 0;
    }
    owned.pub_mgr.next_output_byte = owned.staging.as_mut_ptr();
    owned.pub_mgr.free_in_buffer = owned.staging.len();
    1
}

/// Upstream `term_destination` (jdatadst.c:159-174): flush the tail, `fflush`,
/// and fail on any `ferror` the stream accumulated.
unsafe extern "C" fn stdio_term_destination(cinfo: *mut c_void) {
    let owned: &mut OwnedDestMgr = match unsafe { owned_dest_mut(cinfo) } {
        Some(o) => o,
        None => return,
    };
    let file: *mut c_void = match owned.kind {
        JpegDest::File(ref s) => s.outfile,
        JpegDest::Mem(_) => {
            owned.pending_error = Some(JERR_BUFFER_SIZE);
            return;
        }
    };
    if file.is_null() {
        owned.pending_error = Some(JERR_FILE_WRITE);
        return;
    }
    let datacount: usize = owned
        .staging
        .len()
        .saturating_sub(owned.pub_mgr.free_in_buffer);
    if datacount > 0 && write_c_file(file, &owned.staging[..datacount]) != datacount {
        owned.pending_error = Some(JERR_FILE_WRITE);
        return;
    }
    // SAFETY: `file` is the caller-supplied `FILE *`, still open per the
    // libjpeg contract that the application closes it after compression.
    unsafe {
        c_fflush(file);
        if c_ferror(file) != 0 {
            owned.pending_error = Some(JERR_FILE_WRITE);
        }
    }
}

extern "C" {
    #[link_name = "fwrite"]
    fn c_fwrite(ptr: *const c_void, size: usize, nmemb: usize, stream: *mut c_void) -> usize;
    #[link_name = "fflush"]
    fn c_fflush(stream: *mut c_void) -> c_int;
    #[link_name = "ferror"]
    fn c_ferror(stream: *mut c_void) -> c_int;
}

/// Write `bytes` to a `FILE *` via libc `fwrite`, returning the byte count it
/// accepted so callers can detect a short write.
fn write_c_file(file: *mut c_void, bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    // SAFETY: `file` is a caller-supplied `FILE *` and `bytes` is a live slice.
    unsafe { c_fwrite(bytes.as_ptr() as *const c_void, 1, bytes.len(), file) }
}

/// Whether `cinfo->dest` is the manager this module installed.
fn dest_mgr_is_ours(c: &JpegCompressPublic, priv_state: &CompressPrivate) -> bool {
    match priv_state.dest_mgr {
        Some(ref mgr) => std::ptr::eq(
            c.dest as *const JpegDestinationMgr,
            &mgr.pub_mgr as *const JpegDestinationMgr,
        ),
        None => false,
    }
}

/// Reject reusing a destination manager this function did not create.
///
/// Upstream (jdatadst.c:204-212 / 252-257) compares `init_destination` against
/// its own callback: the manager's private area behind the public struct may be
/// a different size, so reinterpreting a foreign one would read and write out
/// of bounds. We compare the manager's *address* against our own box and check
/// the recorded kind, which decides the same question without depending on
/// function-pointer identity surviving linker identical-code folding.
///
/// Returns `true` when the caller may proceed.
fn dest_mgr_is_reusable(
    cinfo: *mut c_void,
    c: &JpegCompressPublic,
    priv_state: &CompressPrivate,
    want_mem: bool,
) -> bool {
    if c.dest.is_null() {
        return true;
    }
    let compatible: bool = dest_mgr_is_ours(c, priv_state)
        && priv_state
            .dest_mgr
            .as_ref()
            .is_some_and(|mgr| matches!(mgr.kind, JpegDest::Mem(_)) == want_mem);
    if !compatible {
        invoke_error_exit(cinfo, JERR_BUFFER_SIZE);
        return false;
    }
    true
}

/// Install a destination manager that streams bytes into a `FILE *`.
#[no_mangle]
pub extern "C" fn jpeg_stdio_dest(cinfo: *mut c_void, outfile: *mut c_void) {
    crate::unwind_guard!((), {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        let priv_ptr: *mut c_void = c.master;
        let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return,
        };
        if !dest_mgr_is_reusable(cinfo, c, priv_state, /*want_mem=*/ false) {
            return;
        }
        install_dest_mgr(
            c,
            priv_state,
            JpegDest::File(StdioDestState { outfile }),
            stdio_init_destination,
            stdio_empty_output_buffer,
            stdio_term_destination,
        );
    })
}

/// Install a destination manager that fills the caller's buffer, growing into
/// library-allocated memory only when the caller's capacity runs out.
///
/// Mirrors upstream `jpeg_mem_dest` (jdatadst.c:236-277) including its
/// ownership rule: `*outsize` is the caller buffer's **capacity**, and a
/// caller-supplied buffer is never freed by this library.
#[no_mangle]
pub extern "C" fn jpeg_mem_dest(
    cinfo: *mut c_void,
    outbuffer: *mut *mut u8,
    outsize: *mut std::os::raw::c_ulong,
) {
    crate::unwind_guard!((), {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        if outbuffer.is_null() || outsize.is_null() {
            // Upstream: ERREXIT(cinfo, JERR_BUFFER_SIZE) (jdatadst.c:242-243).
            invoke_error_exit(cinfo, JERR_BUFFER_SIZE);
            return;
        }
        let priv_ptr: *mut c_void = c.master;
        let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return,
        };
        if !dest_mgr_is_reusable(cinfo, c, priv_state, /*want_mem=*/ true) {
            return;
        }

        // SAFETY: both pointers are non-null caller storage, checked above.
        let (caller_buffer, caller_capacity): (*mut u8, usize) =
            unsafe { (*outbuffer, *outsize as usize) };
        let mut state: MemDestState = MemDestState {
            outbuffer,
            outsize,
            newbuffer: std::ptr::null_mut(),
            buffer: caller_buffer,
            bufsize: caller_capacity,
        };
        if caller_buffer.is_null() || caller_capacity == 0 {
            // No usable caller buffer: allocate one and publish it right away,
            // exactly as upstream does (jdatadst.c:267-273). Consumers such as
            // libtiff read `*outbuffer` before any compression happens.
            let fresh: *mut u8 = crate::alloc::libc_malloc(OUTPUT_BUF_SIZE);
            if fresh.is_null() {
                invoke_error_exit_parm(cinfo, JERR_OUT_OF_MEMORY, 10);
                return;
            }
            state.newbuffer = fresh;
            state.buffer = fresh;
            state.bufsize = OUTPUT_BUF_SIZE;
            // SAFETY: caller storage, validated non-null above.
            unsafe {
                *outbuffer = fresh;
                *outsize = OUTPUT_BUF_SIZE as std::os::raw::c_ulong;
            }
        }

        let buffer: *mut u8 = state.buffer;
        let bufsize: usize = state.bufsize;
        install_dest_mgr(
            c,
            priv_state,
            JpegDest::Mem(state),
            mem_init_destination,
            mem_empty_output_buffer,
            mem_term_destination,
        );
        if let Some(ref mut mgr) = priv_state.dest_mgr {
            // Upstream publishes the write cursor from `jpeg_mem_dest` itself,
            // not from `init_destination` (jdatadst.c:275-276).
            mgr.pub_mgr.next_output_byte = buffer;
            mgr.pub_mgr.free_in_buffer = bufsize;
        }
    })
}

/// Create-or-reuse the `OwnedDestMgr` and point `cinfo.dest` at its public
/// part.
///
/// Upstream keeps the manager in the permanent pool so it survives
/// `jpeg_abort` and multi-image reuse; our equivalent is the box owned by
/// `CompressPrivate`, which lives until `jpeg_destroy_compress`.
fn install_dest_mgr(
    c: &mut JpegCompressPublic,
    priv_state: &mut CompressPrivate,
    kind: JpegDest,
    init_destination: unsafe extern "C" fn(*mut c_void),
    empty_output_buffer: unsafe extern "C" fn(*mut c_void) -> CBoolean,
    term_destination: unsafe extern "C" fn(*mut c_void),
) {
    match priv_state.dest_mgr {
        Some(ref mut mgr) => {
            mgr.pub_mgr.init_destination = Some(init_destination);
            mgr.pub_mgr.empty_output_buffer = Some(empty_output_buffer);
            mgr.pub_mgr.term_destination = Some(term_destination);
            mgr.kind = kind;
            mgr.pending_error = None;
        }
        None => {
            priv_state.dest_mgr = Some(Box::new(OwnedDestMgr::new(
                kind,
                init_destination,
                empty_output_buffer,
                term_destination,
            )));
        }
    }
    c.dest = priv_state
        .dest_mgr
        .as_mut()
        .map(|mgr| &mut mgr.pub_mgr as *mut JpegDestinationMgr)
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
    crate::unwind_guard!((), {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        // JCS_EXT_* values follow libjpeg-turbo `jmorecfg.h` enum order
        // (`JCS_EXT_RGB = 6` … `JCS_EXT_ARGB = 15`). All RGB-family
        // packed inputs map to a YCbCr-encoded JPEG datastream.
        let jcs: c_int = match c.in_color_space {
            JCS_GRAYSCALE => JCS_GRAYSCALE,
            JCS_RGB => JCS_YCBCR,
            JCS_YCBCR => JCS_YCBCR,
            JCS_CMYK => JCS_CMYK,
            JCS_YCCK => JCS_YCCK,
            6..=15 => JCS_YCBCR,
            _ => JCS_UNKNOWN,
        };
        jpeg_set_colorspace(cinfo, jcs);
    })
}

/// `jpeg_set_colorspace(cinfo, colorspace)` — select the JPEG color space
/// and populate `comp_info[]` with libjpeg's defaults.
#[no_mangle]
pub extern "C" fn jpeg_set_colorspace(cinfo: *mut c_void, colorspace: c_int) {
    crate::unwind_guard!((), {
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
    })
}

/// `jpeg_set_defaults(cinfo)` — populate default compression parameters,
/// mirroring libjpeg `jcparam.c::jpeg_set_defaults`. Requires the caller to
/// have already set `in_color_space`.
#[no_mangle]
pub extern "C" fn jpeg_set_defaults(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
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
    })
}

/// `jpeg_set_quality(cinfo, quality, force_baseline)` — record the requested
/// quality and matching public scale factors. P4-85 tracks installation of
/// the scaled public tables and the `force_baseline` behavior; output currently
/// derives default tables from the private quality value at finish time.
///
/// The scaling curve matches libjpeg `jpeg_quality_scaling`; slots 0 and 1 of
/// `q_scale_factor[]` receive that scale per `jcparam.c::jpeg_set_quality`.
#[no_mangle]
pub extern "C" fn jpeg_set_quality(cinfo: *mut c_void, quality: c_int, _force_baseline: CBoolean) {
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        let priv_ptr: *mut c_void = c.master;
        let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return,
        };
        // When raw_data_in is set, the state machine uses CSTATE_RAW_OK
        // instead of CSTATE_SCANNING so jpeg_write_raw_data knows it is legal.
        c.global_state = if c.raw_data_in != 0 {
            CSTATE_RAW_OK
        } else {
            CSTATE_SCANNING
        };
        c.next_scanline = 0;

        // --- Populate derived fields (libjpeg `jcmaster.c::initial_setup`). ---
        calc_compress_jpeg_dimensions(c, priv_state.lossless_predictor != 0);
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
        // total_iMCU_rows = ceil(image_height / (max_v_samp_factor * data_unit)).
        // `data_unit` is 8 for lossy JPEG and 1 for lossless JPEG.
        let data_unit: u32 = c.min_DCT_v_scaled_size.max(1) as u32;
        let imcu_row_height: u32 = (max_v as u32).saturating_mul(data_unit).max(1);
        c.total_iMCU_rows = c.image_height.div_ceil(imcu_row_height);

        let input_components: usize = c.input_components.max(1) as usize;
        let width: usize = c.image_width as usize;
        let height: usize = c.image_height as usize;
        let row_bytes: usize = width.saturating_mul(input_components);
        let total_bytes: usize = row_bytes.saturating_mul(height);
        priv_state.pixels_u8.clear();
        priv_state.pixels_u16.clear();
        // In raw-data mode we don't pre-allocate a pixel buffer — rows come
        // in via jpeg_write_raw_data instead.
        if c.raw_data_in == 0 {
            priv_state.pixels_u8.resize(total_bytes, 0);
        }
        // Reset raw-data accumulation buffers (both 8-bit and 12-bit).
        priv_state.raw_plane_buffers.clear();
        priv_state.raw_plane_buffers_12.clear();
        priv_state.raw_rows_filled.clear();
        priv_state.raw_plane_widths.clear();
        priv_state.raw_plane_heights.clear();
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
    })
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
    crate::unwind_guard!(0, {
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
    })
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

    let restart_interval: u16 = c.restart_interval.min(u16::MAX as c_uint) as u16;
    let restart_in_rows: u16 = c.restart_in_rows.clamp(0, u16::MAX as c_int) as u16;
    let baseline_restart_interval: u16 = if restart_in_rows > 0 {
        // For a non-progressive interleaved scan, libjpeg converts the public
        // row count into MCUs once during initial_setup. Progressive writers
        // instead retain restart_in_rows and derive the interval per scan.
        let data_unit: u32 = c.min_DCT_h_scaled_size.max(1) as u32;
        // C's master selection disables chroma subsampling for lossless
        // output before per_scan_setup, so a lossless MCU is one sample wide
        // even if jpeg_set_defaults initially installed 2x2 factors.
        let effective_max_h: u32 = if priv_state.lossless_predictor != 0 {
            1
        } else {
            c.max_h_samp_factor.max(1) as u32
        };
        let mcu_width: u32 = effective_max_h.saturating_mul(data_unit).max(1);
        let jpeg_width: u32 = if c.jpeg_width != 0 {
            c.jpeg_width
        } else {
            c.image_width
        };
        let mcus_per_row: u32 = jpeg_width.div_ceil(mcu_width);
        (restart_in_rows as u32)
            .saturating_mul(mcus_per_row)
            .min(u16::MAX as u32) as u16
    } else {
        restart_interval
    };
    let dct_method = libjpeg_turbo_rs::DctMethod::IsLow;

    // Choose encode variant based on parameters captured earlier.
    let bytes_result = if priv_state.lossless_predictor != 0 {
        libjpeg_turbo_rs::encode::pipeline::compress_lossless_extended(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.lossless_predictor,
            priv_state.lossless_point_transform,
            baseline_restart_interval,
        )
    } else if c.progressive_mode != 0 && c.arith_code != 0 {
        libjpeg_turbo_rs::encode::pipeline::compress_arithmetic_progressive(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
            dct_method,
            restart_interval,
            restart_in_rows,
            None,
        )
    } else if c.progressive_mode != 0 {
        libjpeg_turbo_rs::encode::pipeline::compress_progressive_with_restart(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
            dct_method,
            restart_interval,
            restart_in_rows,
            None,
        )
    } else if c.arith_code != 0 {
        libjpeg_turbo_rs::encode::pipeline::compress_arithmetic(
            &priv_state.pixels_u8,
            width,
            height,
            pf,
            priv_state.quality,
            subsamp,
            dct_method,
            baseline_restart_interval,
            None,
        )
    } else if c.optimize_coding != 0 || c.smoothing_factor != 0 {
        libjpeg_turbo_rs::encode::pipeline::compress_optimized_with_params(
            &libjpeg_turbo_rs::encode::pipeline::CompressParams::new(
                &priv_state.pixels_u8,
                width,
                height,
                pf,
                priv_state.quality,
                subsamp,
            )
            .dct_method(dct_method)
            .restart_interval(baseline_restart_interval)
            .smoothing_factor(c.smoothing_factor.clamp(0, 100) as u8)
            .optimize_huffman(c.optimize_coding != 0),
        )
    } else {
        libjpeg_turbo_rs::encode::pipeline::compress_with_params(
            &libjpeg_turbo_rs::encode::pipeline::CompressParams::new(
                &priv_state.pixels_u8,
                width,
                height,
                pf,
                priv_state.quality,
                subsamp,
            )
            .dct_method(dct_method)
            .restart_interval(baseline_restart_interval),
        )
    };

    // Encode + marker injection + push live entirely inside this scope
    // so the encoded `Vec`(s) drop on block exit. `raise_cant_suspend`
    // (called below if the destination signalled suspension via
    // `empty_output_buffer` returning `FALSE`) may `longjmp` past Rust
    // destructors; if any heap-owning local is still alive at that
    // call site, longjmp leaks it. Capture only the boolean status.
    let push_status: Option<bool> = (|| -> Option<bool> {
        let encoded: Vec<u8> = match bytes_result {
            Ok(b) => b,
            Err(e) => {
                priv_state.last_error =
                    CString::new(format!("jpeg_finish_compress: {e}")).unwrap_or_default();
                return None;
            }
        };
        // Inject application-supplied markers right after SOI. The
        // no-marker branch *moves* `encoded` into `with_markers`; the
        // with-marker branch borrows `encoded`, so we explicitly drop
        // it after the borrow ends.
        let with_markers: Vec<u8> =
            if priv_state.pending_markers.is_empty() && priv_state.icc_profile.is_none() {
                encoded
            } else {
                let injected = inject_markers_after_soi(&encoded, priv_state);
                drop(encoded);
                injected
            };
        let ok = push_bytes_through_dest_mgr(c, priv_state, &with_markers);
        drop(with_markers);
        Some(ok)
    })();

    match push_status {
        None => false,
        Some(completed) => {
            let failed: bool = !completed || dest_flush_failed(priv_state);
            finish_dest_flush(c, priv_state, completed);
            !failed
        }
    }
}

/// Emit `encoded` JPEG bytes by writing into the destination manager's
/// `next_output_byte` buffer, invoking `empty_output_buffer` whenever
/// the staging buffer fills.
///
/// Returns `true` on completion (`term_destination` invoked). Returns
/// `false` when the destination manager returned `FALSE` from
/// `empty_output_buffer`; in that case `term_destination` is *not*
/// called and the caller is responsible for dropping any local heap
/// state and then invoking `cinfo->err->error_exit` (per the
/// `JERR_CANT_SUSPEND` contract documented at the call sites). Doing
/// the `error_exit` here directly would `longjmp` past live Rust `Vec`
/// allocations on the caller's stack, leaking them — so the contract
/// is split: this function reports the status, the caller signals it.
///
/// libjpeg.txt §5.5 defines suspension at the `jpeg_write_scanlines`
/// boundary: the entropy coder's `empty_output_buffer` FALSE return
/// propagates up through `process_data` so `jpeg_write_scanlines` can
/// return a row count short of the requested rows. The shim's
/// deferred-encode architecture cannot honor that contract — by the
/// time the bytes reach this function the entire stream is already
/// encoded. The honest response is `JERR_CANT_SUSPEND` (upstream code
/// 25 at `JPEG_LIB_VERSION = 80`, exact message "Suspension not
/// allowed here") — anything else either silently drops bytes (the
/// pre-fix defect) or invents a non-upstream resume contract.
#[must_use = "callers must handle FALSE-return suspension by dropping local state and then invoking error_exit"]
fn push_bytes_through_dest_mgr(
    c: &mut JpegCompressPublic,
    _priv_state: &mut CompressPrivate,
    encoded: &[u8],
) -> bool {
    if c.dest.is_null() {
        return true;
    }
    let mut offset: usize = 0;
    while offset < encoded.len() {
        let need_refill: bool = {
            let dest: &JpegDestinationMgr = unsafe { &*c.dest };
            dest.free_in_buffer == 0 || dest.next_output_byte.is_null()
        };
        if need_refill {
            let empty_fn: Option<unsafe extern "C" fn(*mut c_void) -> CBoolean> =
                unsafe { (*c.dest).empty_output_buffer };
            if let Some(f) = empty_fn {
                let rc: CBoolean = unsafe { f(c as *mut JpegCompressPublic as *mut c_void) };
                if rc == 0 {
                    // Consumer signalled suspension. Skip `term_destination`
                    // and propagate the status to the caller without
                    // calling `error_exit` here (that would `longjmp`
                    // past live Rust `Vec` allocations on the caller's
                    // stack, leaking them).
                    return false;
                }
            } else {
                // No `empty_output_buffer` and no room left. Upstream cannot
                // reach this (the callback is mandatory), and continuing would
                // publish a truncated stream as a success — the exact defect
                // this function's contract forbids. Report it as suspension so
                // the caller raises an error instead.
                return false;
            }
        }
        let (dst_ptr, capacity): (*mut u8, usize) = {
            let dest: &JpegDestinationMgr = unsafe { &*c.dest };
            (dest.next_output_byte, dest.free_in_buffer)
        };
        if dst_ptr.is_null() || capacity == 0 {
            // An application manager returned TRUE from `empty_output_buffer`
            // without actually granting space. Same reasoning as above: never
            // silently drop the remaining bytes.
            return false;
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
    // IMPORTANT: do NOT zero `next_output_byte` / `free_in_buffer`
    // after `term_destination` returns. Pillow's `_imaging.so`
    // `ImagingJpegEncode` computes `bytes_written = state->bytes -
    // free_in_buffer` by reading `cinfo->dest->free_in_buffer`
    // *after* this function returns; clobbering it to 0 made Pillow
    // see a phantom write equal to its full buffer size (or, when
    // its outer wrapper validated the count, fall through to "0
    // bytes" and write an empty file). Leave the destination state
    // exactly as `term_destination` left it so the caller can
    // compute its own byte count.
    let term_fn: Option<unsafe extern "C" fn(*mut c_void)> = unsafe { (*c.dest).term_destination };
    if let Some(f) = term_fn {
        unsafe {
            f(c as *mut JpegCompressPublic as *mut c_void);
        }
    }
    true
}

/// Stash the `JERR_CANT_SUSPEND` diagnostic in `priv_state.last_error`
/// and signal the caller via `cinfo->err->error_exit`. Must be called
/// only AFTER any caller-stack `Vec` allocations have been dropped /
/// moved out of scope, because `error_exit` may `longjmp` and skip
/// Rust destructors.
fn raise_cant_suspend(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) {
    priv_state.last_error = CString::new(
        "destination manager returned FALSE from empty_output_buffer; \
         upstream-style suspension is not supported at the flush boundary \
         — see push_bytes_through_dest_mgr in jpeglib.rs",
    )
    .unwrap_or_default();
    invoke_error_exit(
        c as *mut JpegCompressPublic as *mut c_void,
        JERR_CANT_SUSPEND,
    );
}

/// Whether the library destination manager recorded a failure during the last
/// flush. Peeks without consuming, so `finish_dest_flush` still reports it.
fn dest_flush_failed(priv_state: &CompressPrivate) -> bool {
    priv_state
        .dest_mgr
        .as_ref()
        .is_some_and(|mgr| mgr.pending_error.is_some())
}

/// Consume the failure the library destination manager recorded, if any.
///
/// Safe to reach through `&mut CompressPrivate` because no destination
/// callback is running at this point — see the `OwnedDestMgr` note on why the
/// callbacks themselves must not take this route.
fn take_pending_dest_error(priv_state: &mut CompressPrivate) -> Option<c_int> {
    priv_state
        .dest_mgr
        .as_mut()
        .and_then(|mgr| mgr.pending_error.take())
}

/// Close out a `push_bytes_through_dest_mgr` call: report whatever the
/// destination manager signalled through `cinfo->err->error_exit`.
///
/// Two distinct failures arrive here. A library destination manager that hit
/// an I/O or allocation failure recorded its upstream `JERR_*` code in
/// `pending_dest_error` (it cannot `longjmp` from inside the callback without
/// leaking the encoded buffer — see the field's documentation). An
/// *application-supplied* manager that returned `FALSE` from
/// `empty_output_buffer` is asking for suspension, which the deferred-encode
/// shim cannot honour, so it gets `JERR_CANT_SUSPEND`.
///
/// Like `raise_cant_suspend`, this must run only after the caller's own heap
/// allocations are out of scope: `error_exit` normally `longjmp`s away.
fn finish_dest_flush(
    c: &mut JpegCompressPublic,
    priv_state: &mut CompressPrivate,
    completed: bool,
) {
    if let Some(code) = take_pending_dest_error(priv_state) {
        priv_state.last_error = CString::new(match code {
            JERR_FILE_WRITE => {
                "destination stream rejected the encoded bytes (short write or \
                 stream error at flush) — JERR_FILE_WRITE"
            }
            JERR_OUT_OF_MEMORY => {
                "could not grow the jpeg_mem_dest output buffer — JERR_OUT_OF_MEMORY"
            }
            _ => "destination manager reported a fatal error",
        })
        .unwrap_or_default();
        invoke_error_exit(c as *mut JpegCompressPublic as *mut c_void, code);
        return;
    }
    if !completed {
        raise_cant_suspend(c, priv_state);
    }
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
    crate::unwind_guard!((), {
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
        } else if c.raw_data_in != 0 {
            // Raw-data encode path: flush accumulated per-component planes
            // collected by jpeg_write_raw_data / jpeg12_write_raw_data.
            // Dispatch on `data_precision`: 12-bit goes through the
            // `compress_raw_12` backend (i16 planes), every other precision
            // through the 8-bit `compress_raw` backend (u8 planes).
            if c.data_precision == 12 {
                let _ = run_raw_encoder_12_and_flush(c, priv_state);
            } else {
                let _ = run_raw_encoder_and_flush(c, priv_state);
            }
        } else {
            let _ = run_encoder_and_flush(c, priv_state);
        }
        c.global_state = CSTATE_START;
    })
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
    crate::unwind_guard!(-1, {
        let q: u8 = quality.clamp(1, 100) as u8;
        libjpeg_turbo_rs::quality_scaling(q) as c_int
    })
}

/// `jpeg_add_quant_table(cinfo, which_tbl, basic_table, scale_factor,
///                       force_baseline)`.
///
/// Installs a quantization table at slot `which_tbl` (0..3). `basic_table`
/// is in natural order, 64 entries. `scale_factor` matches libjpeg: 100
/// leaves the table unchanged; smaller values scale toward finer quant,
/// larger toward coarser. When `force_baseline` is non-zero, all entries
/// are clamped to 1..255 for baseline JPEG compatibility.
///
/// The native encoder accepts raw tables, but this classic boundary currently
/// stores them only in private state and does not apply them at
/// `jpeg_finish_compress` time. P4-85 tracks that output no-op plus the public
/// slot and `force_baseline` semantics.
#[no_mangle]
pub extern "C" fn jpeg_add_quant_table(
    cinfo: *mut c_void,
    which_tbl: c_int,
    basic_table: *const u32,
    scale_factor: c_int,
    _force_baseline: CBoolean,
) {
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
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
    })
}

/// `jpeg_simple_progression(cinfo)` — switch the encoder to the default
/// libjpeg progressive scan script.
#[no_mangle]
pub extern "C" fn jpeg_simple_progression(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return,
        };
        c.progressive_mode = 1;
    })
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
    crate::unwind_guard!((), {
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
    })
}

/// `jpeg_suppress_tables(cinfo, suppress)` — when set, quant and Huffman
/// tables are omitted from the next datastream (caller must have emitted
/// them separately via `jpeg_write_tables`). Stored in private state.
#[no_mangle]
pub extern "C" fn jpeg_suppress_tables(cinfo: *mut c_void, suppress: CBoolean) {
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
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
    })
}

/// `jpeg_write_m_byte(cinfo, val)` — append a single byte to the most
/// recently opened marker segment (`jpeg_write_m_header`).
#[no_mangle]
pub extern "C" fn jpeg_write_m_byte(cinfo: *mut c_void, val: c_int) {
    crate::unwind_guard!((), {
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
    })
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
    crate::unwind_guard!((), {
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
    })
}

/// `jpeg_write_tables(cinfo)` — write an abbreviated JPEG datastream
/// containing only quantization / Huffman tables (SOI, DQT, DHT, EOI).
/// Applications call this to share table state across multiple images.
///
/// We emit a standard quality-75 baseline table pair, which matches
/// `jpeg_set_defaults(); jpeg_set_quality(75, TRUE); jpeg_write_tables()`.
#[no_mangle]
pub extern "C" fn jpeg_write_tables(cinfo: *mut c_void) {
    crate::unwind_guard!((), {
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
        // Build a tables-only datastream that matches what upstream
        // `jcmarker.c::write_tables_only` emits:
        //   * Huffman (`arith_code == 0`): SOI + DQT(Tq=0/1) + DHT(DC/AC ×
        //     luma/chroma) + EOI.
        //   * Arithmetic (`arith_code != 0`): SOI + DQT(Tq=0/1) + EOI. No
        //     DHT (upstream skips them for arithmetic) and no DAC (DAC is
        //     emitted later with the scan header for the abbreviated body
        //     stream, not here).
        // Strategy: encode a tiny 16×16 RGB Huffman dummy at 4:2:0 — the
        // resulting DQT segments are identical regardless of entropy mode
        // (quant tables depend only on quality), so we just filter out DHT
        // when arithmetic is requested.
        let arith: bool = c.arith_code != 0;
        let tables_bytes: Vec<u8> = build_tables_only_datastream(priv_state.quality, arith);
        // Push through the destination manager exactly like a full encode.
        if c.dest.is_null() {
            return;
        }
        if let Some(init) = unsafe { (*c.dest).init_destination } {
            unsafe { init(cinfo) };
        }
        let completed: bool = push_bytes_through_dest_mgr(c, priv_state, &tables_bytes);
        drop(tables_bytes);
        finish_dest_flush(c, priv_state, completed);
    })
}

/// Emit a tables-only JPEG datastream at the given quality, matching
/// upstream `jcmarker.c::write_tables_only`:
///
/// * Huffman (`arith == false`): `SOI + DQT(Tq=0/1) + DHT(DC/AC ×
///   luma/chroma) + EOI`.
/// * Arithmetic (`arith == true`): `SOI + DQT(Tq=0/1) + EOI`. No DHT
///   (upstream skips them when `cinfo->arith_code` is set) and no DAC
///   (DAC is emitted later with the scan header for the abbreviated
///   body stream, not as part of the tables-only datastream).
///
/// Strategy: encode a 16×16 RGB Huffman dummy at 4:2:0 so the encoder
/// emits both luma and chroma quantization tables and all four standard
/// Huffman tables, then strip SOF/SOS/scan data. The DQT segments are
/// identical regardless of entropy mode (quant tables depend only on
/// quality), so the same Huffman dummy works for both modes — DHT is
/// just filtered out when `arith` is true. A grayscale-only caller
/// receives extra unused chroma tables — harmless, since downstream
/// abbreviated reads ignore unreferenced indices.
fn build_tables_only_datastream(quality: u8, arith: bool) -> Vec<u8> {
    // Smallest input that still triggers a full color-encode emission
    // of both quantization tables and all four Huffman tables: a single
    // 16×16 (one 4:2:0 MCU) RGB block. Pixel content is irrelevant —
    // tables are derived from quality + Annex K standard tables, not
    // from the data.
    let dummy: Vec<u8> = vec![0u8; 16 * 16 * 3];
    let encoded: Vec<u8> = match libjpeg_turbo_rs::compress(
        &dummy,
        16,
        16,
        PixelFormat::Rgb,
        quality,
        libjpeg_turbo_rs::Subsampling::S420,
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
            // DQT (0xDB) always; DHT (0xC4) only when Huffman. Upstream
            // `write_tables_only` skips DHT for arithmetic-coded
            // compression, and never emits DAC in this datastream
            // (DAC accompanies the scan header, not the tables-only
            // stream).
            0xDB | 0xC4 if code == 0xDB || !arith => {
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
            0xDA => {
                // SOS — actual end of the table-bearing region. Tables are
                // valid only up to here.
                break;
            }
            0xC0..=0xC3 => {
                // SOF — skip the frame header. JPEG file order is
                // SOI → APP/COM → DQT → SOF → DHT → SOS, so DHT segments
                // arrive *after* SOF. If we broke here we would lose the
                // Huffman tables.
                if i + 4 > encoded.len() {
                    break;
                }
                let seg_len: usize = ((encoded[i + 2] as usize) << 8) | (encoded[i + 3] as usize);
                i += 2 + seg_len;
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
    crate::unwind_guard!(0, {
        write_scanlines_highprec(cinfo, scanlines, num_lines, /*precision=*/ 12)
    })
}

/// `jpeg16_write_scanlines(cinfo, scanlines, num_lines) -> JDIMENSION`.
#[no_mangle]
pub extern "C" fn jpeg16_write_scanlines(
    cinfo: *mut c_void,
    scanlines: *mut *mut u16,
    num_lines: JDimension,
) -> JDimension {
    crate::unwind_guard!(0, {
        write_scanlines_highprec(cinfo, scanlines, num_lines, /*precision=*/ 16)
    })
}

// ---------------------------------------------------------------------------
// Raw-data encode entry points (P0-3 follow-on).
//
// `jpeg_write_raw_data` / `jpeg12_write_raw_data` accept pre-downsampled
// component planes from the caller, bypassing color conversion and chroma
// downsampling. Counterpart to `jpeg_read_raw_data` on the decode side.
//
// Design: buffered accumulation mirroring the read-side lazy materialisation
// used by `jpeg_read_raw_data`.  Each call appends one iMCU row of
// per-component rows into `CompressPrivate::raw_plane_buffers`.  After the
// final call the caller invokes `jpeg_finish_compress`, which passes the
// complete accumulated planes to `libjpeg_turbo_rs::compress_raw` and pushes
// the encoded JPEG through the destination manager.
//
// Scope: 8-bit baseline only.  `jpeg12_write_raw_data` sets a JERR_NOTIMPL
// error message and returns 0; lossless raw-data is likewise out of scope.
// ---------------------------------------------------------------------------

/// `jpeg_write_raw_data(cinfo, data, num_lines) -> JDIMENSION`.
///
/// Accepts one iMCU row of pre-downsampled component data.  On the first call
/// the per-component plane buffers are allocated based on the image dimensions
/// and sampling factors read from `cinfo`.  Subsequent calls append rows.
///
/// Returns `max_v_samp_factor * DCTSIZE` (= lines consumed) on success, or 0
/// on error (state set in `last_error`).
///
/// # Limitations
/// - 8-bit (data_precision == 8) baseline only.
/// - Lossless and 12-bit raw-data are out of scope for this entry point.
#[no_mangle]
pub extern "C" fn jpeg_write_raw_data(
    cinfo: *mut c_void,
    data: *mut *mut *mut u8,
    num_lines: JDimension,
) -> JDimension {
    crate::unwind_guard!(0, {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        let priv_ptr: *mut c_void = c.master;
        let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return 0,
        };

        // Only 8-bit supported; 12/16-bit callers should use jpeg12_write_raw_data.
        if c.data_precision != 8 {
            priv_state.last_error = CString::new(
                "jpeg_write_raw_data: JERR_BAD_PRECISION (only data_precision=8 supported)",
            )
            .unwrap_or_default();
            return 0;
        }

        // Must be in RAW_OK state (set by jpeg_start_compress when raw_data_in=1).
        if c.global_state != CSTATE_RAW_OK {
            priv_state.last_error =
                CString::new("jpeg_write_raw_data: JERR_BAD_STATE (call jpeg_start_compress first with raw_data_in=TRUE)")
                    .unwrap_or_default();
            return 0;
        }

        let max_vsf: usize = c.max_v_samp_factor.max(1) as usize;
        let dct_size: usize = c.min_DCT_v_scaled_size.max(8) as usize;
        let lines_per_imcu: JDimension = (max_vsf * dct_size) as JDimension;

        // Validate caller supplied enough rows.
        if num_lines < lines_per_imcu {
            priv_state.last_error = CString::new(format!(
                "jpeg_write_raw_data: JERR_BUFFER_SIZE (num_lines={num_lines} < lines_per_iMCU={lines_per_imcu})"
            ))
            .unwrap_or_default();
            return 0;
        }

        let num_components: usize = c.num_components.max(0) as usize;
        if num_components == 0 || data.is_null() {
            return 0;
        }

        // On first call, allocate the per-component plane buffers.
        if priv_state.raw_plane_buffers.is_empty() {
            let image_width: usize = c.image_width as usize;
            let image_height: usize = c.image_height as usize;
            let max_hsf: usize = c.max_h_samp_factor.max(1) as usize;

            priv_state.raw_plane_buffers = Vec::with_capacity(num_components);
            priv_state.raw_rows_filled = Vec::with_capacity(num_components);
            priv_state.raw_plane_widths = Vec::with_capacity(num_components);
            priv_state.raw_plane_heights = Vec::with_capacity(num_components);

            for comp_idx in 0..num_components {
                // Per-component sampling factors from comp_info.
                let (h_samp, v_samp): (usize, usize) = if comp_idx < priv_state.comp_info.len() {
                    let ci: &JpegComponentInfoCompress = &priv_state.comp_info[comp_idx];
                    (
                        ci.h_samp_factor.max(1) as usize,
                        ci.v_samp_factor.max(1) as usize,
                    )
                } else {
                    (1, 1)
                };

                // MCU-aligned plane width: ceil(image_width * h_samp / max_hsf),
                // rounded up to the next multiple of dct_size.
                let dct_h: usize = c.min_DCT_h_scaled_size.max(8) as usize;
                let raw_w: usize = (image_width * h_samp).div_ceil(max_hsf).div_ceil(dct_h) * dct_h;
                // MCU-aligned plane height: ceil(image_height * v_samp / max_vsf),
                // rounded up to dct_size.
                let raw_h: usize =
                    (image_height * v_samp).div_ceil(max_vsf).div_ceil(dct_size) * dct_size;

                priv_state.raw_plane_buffers.push(vec![0u8; raw_w * raw_h]);
                priv_state.raw_rows_filled.push(0);
                priv_state.raw_plane_widths.push(raw_w);
                priv_state.raw_plane_heights.push(raw_h);
            }
        }

        // For each component copy comp_v_samp_factor * dct_size rows from the
        // caller's JSAMPIMAGE data[comp][row_idx] into the plane buffer.
        for comp_idx in 0..num_components {
            let (v_samp, h_samp): (usize, usize) = if comp_idx < priv_state.comp_info.len() {
                let ci: &JpegComponentInfoCompress = &priv_state.comp_info[comp_idx];
                (
                    ci.v_samp_factor.max(1) as usize,
                    ci.h_samp_factor.max(1) as usize,
                )
            } else {
                (1, 1)
            };
            let max_hsf: usize = c.max_h_samp_factor.max(1) as usize;
            let rows_this_imcu: usize = v_samp * dct_size;
            let plane_width: usize = priv_state.raw_plane_widths[comp_idx];
            let plane_height: usize = priv_state.raw_plane_heights[comp_idx];
            let already_filled: usize = priv_state.raw_rows_filled[comp_idx];

            // Derive the actual sample width for this component (caller's row
            // stride may be wider but we only copy meaningful pixels).
            let image_width: usize = c.image_width as usize;
            let comp_width: usize = (image_width * h_samp).div_ceil(max_hsf);

            // SAFETY: `data` is a JSAMPIMAGE — pointer to `num_components` entries,
            // each entry is a `JSAMPARRAY` (array of row pointers for this component).
            let comp_array: *mut *mut u8 = unsafe { *data.add(comp_idx) };
            if comp_array.is_null() {
                continue;
            }

            for row_in_imcu in 0..rows_this_imcu {
                let dest_row: usize = already_filled + row_in_imcu;
                if dest_row >= plane_height {
                    // Past the MCU-aligned height — discard padding rows.
                    break;
                }
                let src_row_ptr: *const u8 = unsafe { *comp_array.add(row_in_imcu) as *const u8 };
                if src_row_ptr.is_null() {
                    continue;
                }
                let dst_offset: usize = dest_row * plane_width;
                let copy_bytes: usize = comp_width.min(plane_width);
                // SAFETY: src_row_ptr points to `comp_width` valid samples
                // provided by the caller; dst slice is within our allocation.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_row_ptr,
                        priv_state.raw_plane_buffers[comp_idx]
                            .as_mut_ptr()
                            .add(dst_offset),
                        copy_bytes,
                    );
                }
            }

            priv_state.raw_rows_filled[comp_idx] += rows_this_imcu;
        }

        // Advance output scanline by the luma rows consumed this iMCU row.
        c.next_scanline = (c.next_scanline + lines_per_imcu).min(c.image_height);
        lines_per_imcu
    })
}

/// Invoke `libjpeg_turbo_rs::compress_raw` over the raw planes accumulated by
/// `jpeg_write_raw_data` calls and push the result through the destination
/// manager.  Called from `jpeg_finish_compress` on the `raw_data_in` path.
fn run_raw_encoder_and_flush(c: &mut JpegCompressPublic, priv_state: &mut CompressPrivate) -> bool {
    let image_width: usize = c.image_width as usize;
    let image_height: usize = c.image_height as usize;
    if image_width == 0 || image_height == 0 {
        return false;
    }
    if priv_state.raw_plane_buffers.is_empty() {
        priv_state.last_error = CString::new(
            "jpeg_finish_compress: no raw planes accumulated (jpeg_write_raw_data was not called)",
        )
        .unwrap_or_default();
        return false;
    }

    let num_components: usize = priv_state.raw_plane_buffers.len();
    let subsampling: libjpeg_turbo_rs::Subsampling =
        subsampling_from_comp_info(priv_state, num_components as c_int);

    // `compress_raw` requires exact logical plane dimensions, not the
    // MCU-aligned buffer dimensions stored in `raw_plane_widths/heights`.
    // Derive logical dimensions the same way `compress_raw` validates them:
    //   luma  = image_width × image_height
    //   chroma = ceil(image_width / h_samp) × ceil(image_height / v_samp)
    // The plane buffers are wider/taller but contain the correct pixel data
    // in the top-left logical region; `compress_raw` only reads within the
    // declared dimensions, so passing a buffer with stride == raw_plane_width
    // and logical_width <= raw_plane_width is safe (extra columns are never
    // accessed for the row range 0..logical_height).
    let (h_samp_u8, v_samp_u8): (u8, u8) = subsampling.sampling_factors();
    let (h_samp_factor, v_samp_factor): (usize, usize) = (h_samp_u8 as usize, v_samp_u8 as usize);
    let logical_plane_widths: Vec<usize> = (0..num_components)
        .map(|i| {
            if i == 0 || num_components == 1 {
                image_width
            } else {
                image_width.div_ceil(h_samp_factor)
            }
        })
        .collect();
    let logical_plane_heights: Vec<usize> = (0..num_components)
        .map(|i| {
            if i == 0 || num_components == 1 {
                image_height
            } else {
                image_height.div_ceil(v_samp_factor)
            }
        })
        .collect();

    // `compress_raw` requires stride == logical_width for each plane.
    // For non-MCU-aligned images the buffer stride (`raw_plane_width`) is
    // wider than the logical width, so compact each plane into a tightly-
    // packed `logical_width × logical_height` copy before calling.
    // When stride already matches logical width the copy is avoided.
    let compact_planes: Vec<Vec<u8>> = (0..num_components)
        .map(|comp_idx| {
            let raw_w: usize = priv_state.raw_plane_widths[comp_idx];
            let logical_w: usize = logical_plane_widths[comp_idx];
            let logical_h: usize = logical_plane_heights[comp_idx];
            let buf: &[u8] = &priv_state.raw_plane_buffers[comp_idx];

            if raw_w == logical_w {
                // Zero-copy path: first `logical_w * logical_h` bytes are
                // already densely packed.
                let needed: usize = logical_w * logical_h;
                buf[..needed.min(buf.len())].to_vec()
            } else {
                // Compact: extract the logical sub-region row by row.
                let mut compact: Vec<u8> = Vec::with_capacity(logical_w * logical_h);
                for row in 0..logical_h {
                    let row_start: usize = row * raw_w;
                    let row_end: usize = row_start + logical_w;
                    if row_end <= buf.len() {
                        compact.extend_from_slice(&buf[row_start..row_end]);
                    } else {
                        // Partial / missing row — zero-pad.
                        let avail: usize = buf.len().saturating_sub(row_start);
                        compact.extend_from_slice(&buf[row_start..row_start + avail]);
                        compact.resize(compact.len() + (logical_w - avail), 0);
                    }
                }
                compact
            }
        })
        .collect();

    let planes: Vec<&[u8]> = compact_planes.iter().map(|v| v.as_slice()).collect();

    // Encode + push live in this scope so all heap-owning locals
    // (`compact_planes`, `planes`, `encoded`, `with_markers`) are
    // dropped before any `raise_cant_suspend` that may `longjmp`.
    let push_status: Option<bool> = (|| -> Option<bool> {
        let result: Result<Vec<u8>, _> = libjpeg_turbo_rs::compress_raw(
            &planes,
            &logical_plane_widths,
            &logical_plane_heights,
            image_width,
            image_height,
            priv_state.quality,
            subsampling,
        );

        let encoded: Vec<u8> = match result {
            Ok(b) => b,
            Err(e) => {
                priv_state.last_error =
                    CString::new(format!("jpeg_finish_compress (raw): {e}")).unwrap_or_default();
                return None;
            }
        };

        let with_markers: Vec<u8> =
            if priv_state.pending_markers.is_empty() && priv_state.icc_profile.is_none() {
                encoded
            } else {
                let injected = inject_markers_after_soi(&encoded, priv_state);
                drop(encoded);
                injected
            };

        let ok = push_bytes_through_dest_mgr(c, priv_state, &with_markers);
        drop(with_markers);
        Some(ok)
    })();
    drop(planes);
    drop(compact_planes);
    drop(logical_plane_heights);
    drop(logical_plane_widths);

    match push_status {
        None => false,
        Some(completed) => {
            let failed: bool = !completed || dest_flush_failed(priv_state);
            finish_dest_flush(c, priv_state, completed);
            !failed
        }
    }
}

/// 12-bit counterpart to `run_raw_encoder_and_flush`. Invokes
/// `libjpeg_turbo_rs::compress_raw_12` over the i16 planes accumulated
/// by `jpeg12_write_raw_data` calls and pushes the resulting JPEG bytes
/// through the destination manager. Called from `jpeg_finish_compress`
/// when `c.data_precision == 12` and `c.raw_data_in != 0`.
fn run_raw_encoder_12_and_flush(
    c: &mut JpegCompressPublic,
    priv_state: &mut CompressPrivate,
) -> bool {
    let image_width: usize = c.image_width as usize;
    let image_height: usize = c.image_height as usize;
    if image_width == 0 || image_height == 0 {
        return false;
    }
    if priv_state.raw_plane_buffers_12.is_empty() {
        priv_state.last_error = CString::new(
            "jpeg_finish_compress: no raw planes accumulated (jpeg12_write_raw_data was not called)",
        )
        .unwrap_or_default();
        return false;
    }

    let num_components: usize = priv_state.raw_plane_buffers_12.len();
    let subsampling: libjpeg_turbo_rs::Subsampling =
        subsampling_from_comp_info(priv_state, num_components as c_int);

    // Logical (image-content) plane dimensions per the same rules
    // `compress_raw_12` validates: luma at full image size, chroma
    // ceil-divided by sampling factors. Plane buffers are MCU-aligned
    // (wider/taller); the logical region is a top-left subrectangle.
    let (h_samp_u8, v_samp_u8): (u8, u8) = subsampling.sampling_factors();
    let (h_samp_factor, v_samp_factor): (usize, usize) = (h_samp_u8 as usize, v_samp_u8 as usize);
    let logical_plane_widths: Vec<usize> = (0..num_components)
        .map(|i| {
            if i == 0 || num_components == 1 {
                image_width
            } else {
                image_width.div_ceil(h_samp_factor)
            }
        })
        .collect();
    let logical_plane_heights: Vec<usize> = (0..num_components)
        .map(|i| {
            if i == 0 || num_components == 1 {
                image_height
            } else {
                image_height.div_ceil(v_samp_factor)
            }
        })
        .collect();

    // `compress_raw_12` requires stride == logical_width per plane.
    // Compact each MCU-aligned i16 plane into a tightly-packed
    // `logical_width × logical_height` copy when needed.
    let compact_planes: Vec<Vec<i16>> = (0..num_components)
        .map(|comp_idx| {
            let raw_w: usize = priv_state.raw_plane_widths[comp_idx];
            let logical_w: usize = logical_plane_widths[comp_idx];
            let logical_h: usize = logical_plane_heights[comp_idx];
            let buf: &[i16] = &priv_state.raw_plane_buffers_12[comp_idx];

            if raw_w == logical_w {
                let needed: usize = logical_w * logical_h;
                buf[..needed.min(buf.len())].to_vec()
            } else {
                let mut compact: Vec<i16> = Vec::with_capacity(logical_w * logical_h);
                for row in 0..logical_h {
                    let row_start: usize = row * raw_w;
                    let row_end: usize = row_start + logical_w;
                    if row_end <= buf.len() {
                        compact.extend_from_slice(&buf[row_start..row_end]);
                    } else {
                        let avail: usize = buf.len().saturating_sub(row_start);
                        compact.extend_from_slice(&buf[row_start..row_start + avail]);
                        compact.resize(compact.len() + (logical_w - avail), 0);
                    }
                }
                compact
            }
        })
        .collect();

    let planes: Vec<&[i16]> = compact_planes.iter().map(|v| v.as_slice()).collect();

    // Encode + push live in this scope so all heap-owning locals are
    // dropped before any `raise_cant_suspend` that may longjmp.
    let push_status: Option<bool> = (|| -> Option<bool> {
        let result: Result<Vec<u8>, _> = libjpeg_turbo_rs::raw_data_12::compress_raw_12(
            &planes,
            &logical_plane_widths,
            &logical_plane_heights,
            image_width,
            image_height,
            priv_state.quality,
            subsampling,
        );

        let encoded: Vec<u8> = match result {
            Ok(b) => b,
            Err(e) => {
                priv_state.last_error =
                    CString::new(format!("jpeg_finish_compress (raw 12-bit): {e}"))
                        .unwrap_or_default();
                return None;
            }
        };

        let with_markers: Vec<u8> =
            if priv_state.pending_markers.is_empty() && priv_state.icc_profile.is_none() {
                encoded
            } else {
                let injected = inject_markers_after_soi(&encoded, priv_state);
                drop(encoded);
                injected
            };

        let ok = push_bytes_through_dest_mgr(c, priv_state, &with_markers);
        drop(with_markers);
        Some(ok)
    })();
    drop(planes);
    drop(compact_planes);
    drop(logical_plane_heights);
    drop(logical_plane_widths);

    match push_status {
        None => false,
        Some(completed) => {
            let failed: bool = !completed || dest_flush_failed(priv_state);
            finish_dest_flush(c, priv_state, completed);
            !failed
        }
    }
}

/// `jpeg12_write_raw_data(cinfo, data, num_lines) -> JDIMENSION`.
///
/// 12-bit counterpart to `jpeg_write_raw_data` (libjpeg.txt §15). The
/// caller supplies one iMCU row of 12-bit samples per component as a
/// `*mut *mut *mut i16` (`JSAMPIMAGE16` analogue): outer pointer indexes
/// `num_components` entries; each entry is an array of row pointers
/// (`v_samp_factor * DCTSIZE` rows for that component); each row is a
/// stride of i16 samples in the range 0..4095. The shim copies the
/// rows into per-component plane buffers held in
/// `CompressPrivate::raw_plane_buffers_12`; on `jpeg_finish_compress`
/// the buffers are flushed through `libjpeg_turbo_rs::compress_raw_12`.
///
/// State precondition: `c.global_state == CSTATE_RAW_OK` (set by
/// `jpeg_start_compress` when `c.raw_data_in == TRUE`) and
/// `c.data_precision == 12`. Returns 0 on a precondition or buffer-size
/// mismatch (after populating `priv_state.last_error`); returns
/// `lines_per_iMCU` on success.
#[no_mangle]
pub extern "C" fn jpeg12_write_raw_data(
    cinfo: *mut c_void,
    data: *mut *mut *mut i16,
    num_lines: JDimension,
) -> JDimension {
    crate::unwind_guard!(0, {
        let c: &mut JpegCompressPublic = match unsafe { cinfo_compress_mut(cinfo) } {
            Some(c) => c,
            None => return 0,
        };
        let priv_ptr: *mut c_void = c.master;
        let priv_state: &mut CompressPrivate = match unsafe { priv_compress_from_ptr(priv_ptr) } {
            Some(p) => p,
            None => return 0,
        };

        // 12-bit gate: only `data_precision == 12` reaches this entry point;
        // 8/16-bit callers should use `jpeg_write_raw_data` /
        // `jpeg16_write_raw_data` (the latter is not implemented here).
        if c.data_precision != 12 {
            priv_state.last_error = CString::new(format!(
                "jpeg12_write_raw_data: JERR_BAD_PRECISION (data_precision={} not 12)",
                c.data_precision
            ))
            .unwrap_or_default();
            return 0;
        }

        // Must be in RAW_OK state (set by jpeg_start_compress when raw_data_in=1).
        if c.global_state != CSTATE_RAW_OK {
            priv_state.last_error =
                CString::new("jpeg12_write_raw_data: JERR_BAD_STATE (call jpeg_start_compress first with raw_data_in=TRUE)")
                    .unwrap_or_default();
            return 0;
        }

        let max_vsf: usize = c.max_v_samp_factor.max(1) as usize;
        let dct_size: usize = c.min_DCT_v_scaled_size.max(8) as usize;
        let lines_per_imcu: JDimension = (max_vsf * dct_size) as JDimension;

        if num_lines < lines_per_imcu {
            priv_state.last_error = CString::new(format!(
                "jpeg12_write_raw_data: JERR_BUFFER_SIZE (num_lines={num_lines} < lines_per_iMCU={lines_per_imcu})"
            ))
            .unwrap_or_default();
            return 0;
        }

        let num_components: usize = c.num_components.max(0) as usize;
        if num_components == 0 || data.is_null() {
            return 0;
        }

        // On first call, allocate the per-component plane buffers (i16
        // storage). `raw_rows_filled` / `raw_plane_widths` /
        // `raw_plane_heights` are shared with the 8-bit accumulator, so
        // populate them on the same allocation step.
        if priv_state.raw_plane_buffers_12.is_empty() {
            let image_width: usize = c.image_width as usize;
            let image_height: usize = c.image_height as usize;
            let max_hsf: usize = c.max_h_samp_factor.max(1) as usize;

            priv_state.raw_plane_buffers_12 = Vec::with_capacity(num_components);
            priv_state.raw_rows_filled = Vec::with_capacity(num_components);
            priv_state.raw_plane_widths = Vec::with_capacity(num_components);
            priv_state.raw_plane_heights = Vec::with_capacity(num_components);

            for comp_idx in 0..num_components {
                let (h_samp, v_samp): (usize, usize) = if comp_idx < priv_state.comp_info.len() {
                    let ci: &JpegComponentInfoCompress = &priv_state.comp_info[comp_idx];
                    (
                        ci.h_samp_factor.max(1) as usize,
                        ci.v_samp_factor.max(1) as usize,
                    )
                } else {
                    (1, 1)
                };

                let dct_h: usize = c.min_DCT_h_scaled_size.max(8) as usize;
                let raw_w: usize = (image_width * h_samp).div_ceil(max_hsf).div_ceil(dct_h) * dct_h;
                let raw_h: usize =
                    (image_height * v_samp).div_ceil(max_vsf).div_ceil(dct_size) * dct_size;

                priv_state
                    .raw_plane_buffers_12
                    .push(vec![0i16; raw_w * raw_h]);
                priv_state.raw_rows_filled.push(0);
                priv_state.raw_plane_widths.push(raw_w);
                priv_state.raw_plane_heights.push(raw_h);
            }
        }

        // Copy `v_samp * dct_size` rows per component from caller's
        // `*mut *mut *mut i16` into the i16 plane buffer.
        for comp_idx in 0..num_components {
            let (v_samp, h_samp): (usize, usize) = if comp_idx < priv_state.comp_info.len() {
                let ci: &JpegComponentInfoCompress = &priv_state.comp_info[comp_idx];
                (
                    ci.v_samp_factor.max(1) as usize,
                    ci.h_samp_factor.max(1) as usize,
                )
            } else {
                (1, 1)
            };
            let max_hsf: usize = c.max_h_samp_factor.max(1) as usize;
            let rows_this_imcu: usize = v_samp * dct_size;
            let plane_width: usize = priv_state.raw_plane_widths[comp_idx];
            let plane_height: usize = priv_state.raw_plane_heights[comp_idx];
            let already_filled: usize = priv_state.raw_rows_filled[comp_idx];

            let image_width: usize = c.image_width as usize;
            let comp_width: usize = (image_width * h_samp).div_ceil(max_hsf);

            // SAFETY: `data` is a JSAMPIMAGE16 — pointer to `num_components`
            // entries, each entry is an array of row pointers for that
            // component. Caller asserts the per-component array carries at
            // least `rows_this_imcu` row pointers and each row carries at
            // least `comp_width` valid i16 samples.
            let comp_array: *mut *mut i16 = unsafe { *data.add(comp_idx) };
            if comp_array.is_null() {
                continue;
            }

            for row_in_imcu in 0..rows_this_imcu {
                let dest_row: usize = already_filled + row_in_imcu;
                if dest_row >= plane_height {
                    break;
                }
                let src_row_ptr: *const i16 = unsafe { *comp_array.add(row_in_imcu) as *const i16 };
                if src_row_ptr.is_null() {
                    continue;
                }
                let dst_offset: usize = dest_row * plane_width;
                let copy_samples: usize = comp_width.min(plane_width);
                // SAFETY: src_row_ptr points to `comp_width` valid i16s;
                // dst slice is within our allocation. `copy_nonoverlapping`
                // takes a count in *items*, not bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_row_ptr,
                        priv_state.raw_plane_buffers_12[comp_idx]
                            .as_mut_ptr()
                            .add(dst_offset),
                        copy_samples,
                    );
                }
            }

            priv_state.raw_rows_filled[comp_idx] += rows_this_imcu;
        }

        c.next_scanline = (c.next_scanline + lines_per_imcu).min(c.image_height);
        lines_per_imcu
    })
}

/// `jpeg_set_linear_quality(cinfo, scale_factor, force_baseline)`.
///
/// Applies a linear quality scale factor to the default quant tables,
/// where `scale_factor=100` is "1.0×". Upstream applies the factor
/// directly — `q = (basic_table[i] * scale_factor + 50) / 100`,
/// clamped — so two consecutive scale factors (e.g. 99 vs 100)
/// produce slightly different quant tables.
///
/// Our high-level Rust encoder API takes `quality: u8` (1..100) and
/// applies the same standard nonlinear UI scaling that
/// `jpeg_quality_scaling` produces. Going through a quality round-trip
/// quantises the input scale factor to whichever UI step it lands
/// nearest, which is detectably wrong for callers that pass arbitrary
/// integers. So instead we drive `jpeg_add_quant_table` directly with
/// the standard tables (matching the layout `jpeg_default_qtables`
/// installs upstream) so the registered factor is preserved exactly,
/// and only fall back to `jpeg_set_quality(50)` semantics when the
/// caller asks for `scale_factor == 100` (the "use defaults
/// unscaled" case where both paths agree).
#[no_mangle]
pub extern "C" fn jpeg_set_linear_quality(
    cinfo: *mut c_void,
    scale_factor: c_int,
    force_baseline: CBoolean,
) {
    crate::unwind_guard!((), {
        // Standard luminance and chrominance quant tables, in zig-zag order
        // (matches `references/libjpeg-turbo/src/jcparam.c`'s
        // `std_luminance_quant_tbl[]` and `std_chrominance_quant_tbl[]`,
        // re-shuffled into zig-zag order via `jpeg_zigzag_order`).
        const STD_LUM: [u32; 64] = [
            16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49,
            35, 37, 29, 40, 58, 51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56,
            80, 109, 81, 87, 95, 98, 103, 104, 103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103,
            99,
        ];
        const STD_CHROM: [u32; 64] = [
            17, 18, 18, 24, 21, 24, 47, 26, 26, 47, 99, 66, 56, 66, 99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        ];

        let scale: c_int = scale_factor.max(1);
        // Slot 0 = luminance, slot 1 = chrominance (matches upstream
        // `jpeg_default_qtables`).
        jpeg_add_quant_table(cinfo, 0, STD_LUM.as_ptr(), scale, force_baseline);
        jpeg_add_quant_table(cinfo, 1, STD_CHROM.as_ptr(), scale, force_baseline);
    })
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
    crate::unwind_guard!((), {
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

        // Mirror upstream `transencode_master_selection → jinit_c_master_control →
        // initial_setup` for the subset of derived fields that transupp's
        // `jtransform_execute_transformation` reads off the dst comp_info:
        // `width_in_blocks` and `height_in_blocks`, plus the dst-side
        // `max_h_samp_factor / max_v_samp_factor` and `total_iMCU_rows`.
        // Without these, `do_rot_90 / do_transpose / …` see height_in_blocks=0
        // and skip writing the dst arrays entirely.
        populate_dst_block_dims_for_transform(c);

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
    })
}

/// Populate the per-component `width_in_blocks / height_in_blocks` and
/// the cinfo-level `max_h_samp_factor / max_v_samp_factor /
/// total_iMCU_rows` from `jpeg_width / jpeg_height` and each
/// component's sampling factor. Matches `jcmaster.c::initial_setup`'s
/// transcode-only path (the fields transupp's transform helpers
/// dereference).
fn populate_dst_block_dims_for_transform(c: &mut JpegCompressPublic) {
    if c.comp_info.is_null() || c.num_components <= 0 {
        return;
    }
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
    // Prefer `jpeg_width / jpeg_height` when set; fall back to the
    // 8.x baseline `image_width / image_height`. transupp's
    // `transpose_critical_parameters` swaps `image_width / image_height`
    // for ROT_90/TRANSPOSE, so by the time we get here those dims
    // already reflect the post-transform geometry.
    let w: u32 = if c.jpeg_width != 0 {
        c.jpeg_width
    } else {
        c.image_width
    };
    let h: u32 = if c.jpeg_height != 0 {
        c.jpeg_height
    } else {
        c.image_height
    };
    let denom_w: u64 = (max_h as u64) * 8;
    let denom_h: u64 = (max_v as u64) * 8;
    if denom_w == 0 || denom_h == 0 {
        return;
    }
    // Logical block count (matches stock libjpeg's
    // `jcmaster.c::initial_setup`): the public `comp_info` fields
    // report the number of *real* blocks of coefficient data, NOT
    // the MCU-padded virtual-array extent. C callers reading
    // `dstinfo->comp_info` after `jpeg_write_coefficients` rely on
    // this to know how many real coefficients are present (vs
    // dummy edge blocks the encoder pads on emit).
    //
    // The encoder iterates MCU-aligned bounds internally — the
    // foreign-array materialise path in
    // `materialize_foreign_coef_arrays` derives that padded extent
    // from `iMCU_count * samp_factor` separately so we don't have
    // to lie in `comp_info`.
    for comp in comps.iter_mut() {
        let h_samp: u64 = comp.h_samp_factor.max(1) as u64;
        let v_samp: u64 = comp.v_samp_factor.max(1) as u64;
        comp.width_in_blocks = (w as u64 * h_samp).div_ceil(denom_w) as JDimension;
        comp.height_in_blocks = (h as u64 * v_samp).div_ceil(denom_h) as JDimension;
        comp.dct_h_scaled_size = 8;
        comp.dct_v_scaled_size = 8;
    }
    c.total_iMCU_rows = (h as u64).div_ceil(denom_h) as JDimension;
}

/// Read coefficient blocks out of a foreign `jvirt_barray_ptr*` array
/// (the destination workspace allocated by `jtransform_adjust_parameters`)
/// and assemble them into a `JpegCoefficients` ready for re-encoding.
///
/// The returned coefficients carry only the fields the encoder reads
/// (dimensions, components, quant tables, restart, density). Adobe
/// transform classification is intentionally left as `None` — the
/// destination cinfo's `write_Adobe_marker` flag and `jpeg_color_space`
/// drive the existing 4-component CMYK/YCCK branch in
/// `run_coefficient_writer_and_flush` to inject Adobe APP14 with the
/// right transform byte, and any source APP14 transupp wanted to
/// preserve has already been queued onto `pending_markers` via
/// `jcopy_markers_execute → jpeg_write_marker`.
fn materialize_foreign_coef_arrays(
    c: &JpegCompressPublic,
    handle: *const c_void,
) -> Result<libjpeg_turbo_rs::JpegCoefficients, String> {
    if c.num_components <= 0 {
        return Err("jpeg_finish_compress: dst cinfo has num_components <= 0; \
             call jpeg_copy_critical_parameters before jpeg_write_coefficients"
            .to_string());
    }
    let n: usize = c.num_components as usize;
    if c.comp_info.is_null() {
        return Err("jpeg_finish_compress: dst cinfo has NULL comp_info; \
             call jpeg_copy_critical_parameters before jpeg_write_coefficients"
            .to_string());
    }
    if c.mem.is_null() {
        return Err("jpeg_finish_compress: dst cinfo->mem is NULL".to_string());
    }
    let mem: &memmgr::JpegMemoryMgr = unsafe { &*(c.mem as *const memmgr::JpegMemoryMgr) };
    let access_virt_barray: memmgr::AccessVirtBarrayFn = mem
        .access_virt_barray
        .ok_or("jpeg_finish_compress: access_virt_barray not wired")?;

    let comp_info_slice: &[JpegComponentInfoPublic] =
        unsafe { std::slice::from_raw_parts(c.comp_info, n) };

    // Pull quant tables. Re-emit in zigzag order to match what the
    // Rust encoder expects — `JpegCoefficients::quant_tables` is
    // documented as zigzag (see the writer in
    // `src/api/coefficient.rs::write_coefficients`). Stock libjpeg
    // stores quantval in zigzag order too, so a verbatim copy is
    // already in the right order.
    let mut quant_tables: Vec<[u16; 64]> = Vec::with_capacity(4);
    for slot in c.quant_tbl_ptrs.iter() {
        if slot.is_null() {
            quant_tables.push([0u16; 64]);
        } else {
            let q: &JQuantTblPublic = unsafe { &**slot };
            quant_tables.push(q.quantval);
        }
    }
    // Trim trailing all-zero tables (we want one-per-distinct-table-no
    // entries, not always 4). Find the highest used quant_tbl_no.
    let highest_used: usize = comp_info_slice
        .iter()
        .map(|ci| ci.quant_tbl_no.max(0) as usize)
        .max()
        .unwrap_or(0);
    quant_tables.truncate(highest_used + 1);

    // Cinfo pointer for `access_virt_barray` (it ignores the cinfo
    // arg in our memmgr but keep ABI shape).
    let cinfo_ptr: *mut c_void = c as *const JpegCompressPublic as *mut c_void;
    let array_slot: *const *mut memmgr::JVirtBarrayControl =
        handle as *const *mut memmgr::JVirtBarrayControl;

    // Recover blocks_x / blocks_y per component. The encoder
    // iterates MCU-aligned bounds, NOT the logical
    // `comp_info[ci].width_in_blocks` (which is intentionally
    // logical — see `populate_dst_block_dims_for_transform`).
    // Compute MCU-aligned per-component block counts inline:
    //
    //   blocks_x = ceil(image_width  / (max_h * 8)) * h_samp
    //   blocks_y = ceil(image_height / (max_v * 8)) * v_samp
    //
    // This matches `transupp.c::jtransform_request_workspace`,
    // so for ops that allocated a fresh workspace barray the
    // computed value equals `barray.blocksperrow`. For no-workspace
    // ops where dst aliases the source's full barray
    // (`-flip horizontal`, `-crop +0+0`, etc.), the source
    // barray's extent overshoots the cropped/transformed dst
    // geometry — capping at the MCU-aligned dst count emits only
    // the cropped subset (caught by round-9 verification of
    // `-crop 16x16+0+0`).
    let max_h: u32 = (c.max_h_samp_factor.max(1)) as u32;
    let max_v: u32 = (c.max_v_samp_factor.max(1)) as u32;
    let img_w: u32 = if c.jpeg_width != 0 {
        c.jpeg_width
    } else {
        c.image_width
    };
    let img_h: u32 = if c.jpeg_height != 0 {
        c.jpeg_height
    } else {
        c.image_height
    };
    let imcu_w: u32 = img_w.div_ceil(max_h * 8);
    let imcu_h: u32 = img_h.div_ceil(max_v * 8);
    let mut components: Vec<libjpeg_turbo_rs::ComponentCoefficients> = Vec::with_capacity(n);
    for (ci, comp_info) in comp_info_slice.iter().enumerate() {
        // SAFETY: `array_slot[ci]` is a `JVirtBarrayControl*` placed
        // there by `jtransform_request_workspace` (or
        // `jpeg_read_coefficients` if no workspace was needed); both
        // paths use the cinfo's `request_virt_barray`, so the value
        // lives in our memory pool.
        let barray: *mut memmgr::JVirtBarrayControl = unsafe { *array_slot.add(ci) };
        if barray.is_null() {
            return Err(format!(
                "jpeg_finish_compress: foreign coef_arrays[{ci}] is NULL"
            ));
        }
        let (full_x, full_y): (u32, u32) =
            unsafe { ((*barray).blocksperrow, (*barray).rows_in_array) };
        let h_samp: u32 = (comp_info.h_samp_factor.max(1)) as u32;
        let v_samp: u32 = (comp_info.v_samp_factor.max(1)) as u32;

        // Logical extent — what transupp's `do_*` helpers actually
        // wrote into the workspace. For workspace transforms
        // (`-rotate`, `-transpose`, etc.) this is `comp_info[ci]
        // .{width,height}_in_blocks` as set by
        // `populate_dst_block_dims_for_transform`. For the
        // no-workspace case where dst aliases the source's full
        // barray, the logical extent reflects the cropped/transformed
        // dst geometry, smaller than `full_x / full_y`. We must NOT
        // read past this — `jtransform_request_workspace` allocates
        // the workspace with `pre_zero=FALSE`, so positions beyond
        // the logical extent contain whatever `pool.push_block`
        // returned (uninit Rust allocator memory) — codex round-11
        // flagged this as UB.
        let log_x: u32 = if comp_info.width_in_blocks != 0 {
            comp_info.width_in_blocks.min(full_x)
        } else {
            full_x
        };
        let log_y: u32 = if comp_info.height_in_blocks != 0 {
            comp_info.height_in_blocks.min(full_y)
        } else {
            full_y
        };

        // MCU-padded extent — what stock libjpeg's encoder iterates.
        // For non-MCU-aligned image dimensions, this exceeds the
        // logical extent by up to one column/row of dummy blocks
        // per component. The encoder consumes
        // `JpegCoefficients::blocks` indexed as
        // `[by * blocks_x + bx]`, so we ship MCU-padded dims and
        // zero-pad the trailing region in the materialised Vec.
        let mcu_x_raw: u32 = imcu_w * h_samp;
        let mcu_y_raw: u32 = imcu_h * v_samp;
        let blocks_x: u32 = if mcu_x_raw != 0 {
            mcu_x_raw.max(log_x)
        } else {
            log_x
        };
        let blocks_y: u32 = if mcu_y_raw != 0 {
            mcu_y_raw.max(log_y)
        } else {
            log_y
        };

        let row_array: memmgr::JBlockArray = unsafe {
            access_virt_barray(cinfo_ptr, barray, 0, log_y, /*writable=*/ 0)
        };
        if row_array.is_null() {
            return Err(format!(
                "jpeg_finish_compress: access_virt_barray(comp={ci}) returned NULL"
            ));
        }

        // Initialise to zeros so the MCU-padded trailing column /
        // row is dummy (DC=0; the encoder's
        // `is_dummy → DC=prev_dc[ci]` branch then takes over).
        let total: usize = blocks_x as usize * blocks_y as usize;
        let mut blocks: Vec<[i16; 64]> = vec![[0i16; 64]; total];
        for r in 0..log_y as usize {
            // SAFETY: `access_virt_barray(start_row=0, num_rows=log_y)`
            // returned a contiguous run of `log_y` row pointers; each
            // row points at `full_x >= log_x` `JBlock`s (allocated by
            // `alloc_barray_impl`). We only read the first `log_x`
            // columns to stay inside what transupp initialised.
            let row_ptr: memmgr::JBlockRow = unsafe { *row_array.add(r) };
            let row_blocks: &[memmgr::JBlock] =
                unsafe { std::slice::from_raw_parts(row_ptr, log_x as usize) };
            // Foreign-array entries are populated in natural
            // row-major order (transupp's transformations index
            // coefficients as `block[i*8+j]`). Our encoder consumes
            // `JpegCoefficients.blocks` in zigzag order, so re-pack
            // each block here. Symmetric counterpart of the
            // `NATURAL_ORDER`-based zigzag→natural copy in
            // `jpeg_read_coefficients`.
            for (col, natural_block) in row_blocks.iter().enumerate() {
                let mut zigzag_block: [i16; 64] = [0; 64];
                for (natural_pos, &coef) in natural_block.iter().enumerate() {
                    let zigzag_pos: usize =
                        libjpeg_turbo_rs::common::quant_table::NATURAL_ORDER[natural_pos];
                    zigzag_block[zigzag_pos] = coef;
                }
                blocks[r * blocks_x as usize + col] = zigzag_block;
            }
        }
        components.push(libjpeg_turbo_rs::ComponentCoefficients {
            blocks,
            blocks_x: blocks_x as usize,
            blocks_y: blocks_y as usize,
            h_sampling: comp_info.h_samp_factor.max(1) as u8,
            v_sampling: comp_info.v_samp_factor.max(1) as u8,
            quant_table_index: comp_info.quant_tbl_no.max(0) as u8,
            component_id: comp_info.component_id.max(0) as u8,
        });
    }

    // Decide which dimension fields to use. `jpeg_width / jpeg_height`
    // (libjpeg 8+) reflect any IDCT scaling applied by the source
    // decompressor; transupp updates these to the output dims. For
    // pre-libjpeg-7 callers `image_width / image_height` is the
    // canonical home — fall back to it when jpeg_width/jpeg_height
    // is zero.
    let width: u32 = if c.jpeg_width != 0 {
        c.jpeg_width
    } else {
        c.image_width
    };
    let height: u32 = if c.jpeg_height != 0 {
        c.jpeg_height
    } else {
        c.image_height
    };

    // Preserve the source's Adobe APP14 transform classification when
    // we can recover it AND when the destination output keeps a
    // colourspace where APP14 is meaningful (3 or 4 components).
    //
    // The encoder's 4-component CMYK/YCCK branch injects an Adobe
    // APP14 with the right transform byte regardless, but for
    // 3-component sources that DID carry an Adobe APP14 (e.g. an
    // RGB-encoded JPEG that wants to be re-emitted with
    // `transform=0` to keep the RGB classification) the field is
    // load-bearing because `inject_adobe_app14_after_jfif` only
    // fires when `adobe_transform.is_some()`.
    //
    // For 1-component grayscale outputs (e.g. transupp's
    // `-grayscale` no-workspace path leaves the dst cinfo at
    // `num_components = 1` while still handing us the registered
    // 3-component source array), copying the source's transform
    // would emit a stale APP14 marker that libjpeg's parser would
    // suppress on read — flagged in codex round-8 review of b7f690d.
    //
    // Pull from the side-table CoefHandle when the array was
    // registered by *this* shim's `jpeg_read_coefficients`. Foreign
    // workspace arrays from transupp's `-rotate` etc. preserve
    // Adobe via `jcopy_markers_execute → jpeg_write_marker`
    // instead, so falling through to `None` in that case is
    // correct.
    let adobe_transform: Option<u8> = if n == 3 || n == 4 {
        match coef_lookup_handle(handle) {
            Some(handle_ptr) => unsafe { (*handle_ptr).inner.adobe_transform },
            None => None,
        }
    } else {
        None
    };
    let saw_jfif_marker: bool = match coef_lookup_handle(handle) {
        Some(handle_ptr) => unsafe { (*handle_ptr).inner.saw_jfif_marker },
        None => c.write_JFIF_header != 0,
    };

    Ok(libjpeg_turbo_rs::JpegCoefficients {
        width: width.min(u16::MAX as u32) as u16,
        height: height.min(u16::MAX as u32) as u16,
        data_precision: c.data_precision.clamp(0, 16) as u8,
        components,
        quant_tables,
        restart_interval: c.restart_interval.min(65535) as u16,
        density_unit: c.density_unit,
        x_density: c.X_density,
        y_density: c.Y_density,
        saw_jfif_marker,
        adobe_transform,
    })
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
    // Two valid handle shapes are accepted:
    //
    // 1. **In-process round trip** — the caller passed back the exact
    //    pointer returned from this shim's `jpeg_read_coefficients`.
    //    The pointer is registered in `coef_array_to_handle_table`
    //    against the parsed `CoefHandle`. We pull the cached
    //    `JpegCoefficients` and skip the per-barray re-read.
    //
    // 2. **Foreign virtual coefficient array** (the stock `transupp` /
    //    `jtransform_adjust_parameters` path used by `jpegtran`) — the
    //    pointer is a `jvirt_barray_ptr *` allocated through some
    //    cinfo's `mem` manager, with one entry per component.
    //    Materialise a fresh `JpegCoefficients` by walking the dst
    //    cinfo's `comp_info[]` and reading each barray via
    //    `cinfo->mem->access_virt_barray`. Quant tables, dimensions,
    //    restart, density, and Adobe metadata come from the dst cinfo
    //    state set up by `jpeg_copy_critical_parameters`,
    //    `jtransform_adjust_parameters`, and `jcopy_markers_execute`.
    // Decide which path materialises the JpegCoefficients:
    //
    // * If dst cinfo already has its compress-side metadata wired up
    //   (`num_components > 0` and non-NULL `comp_info`), prefer the
    //   foreign-array materialise path — even when the handle
    //   pointer happens to be in our side table. transupp's
    //   no-workspace transforms (`-flip h`, `-grayscale`) hand
    //   `jpeg_write_coefficients` the **same** pointer that came
    //   out of `jpeg_read_coefficients` and then mutate those
    //   arrays in place during `jtransform_execute_transformation`.
    //   Cloning the cached `CoefHandle` would emit the original
    //   (unmodified) source — caught by codex review of 809f52a.
    //
    // * Otherwise (the in-process round trip without a metadata
    //   copy, e.g. `write_coefficients_roundtrip_pixel_exact`), the
    //   side-table shortcut returns the cached `JpegCoefficients`
    //   and skips the per-barray re-read.
    let dst_has_metadata: bool = c.num_components > 0 && !c.comp_info.is_null() && !c.mem.is_null();
    let mut adjusted: libjpeg_turbo_rs::JpegCoefficients = if dst_has_metadata {
        match materialize_foreign_coef_arrays(c, handle) {
            Ok(coeffs) => coeffs,
            Err(msg) => {
                priv_state.last_error = CString::new(msg).unwrap_or_default();
                return false;
            }
        }
    } else {
        match coef_lookup_handle(handle) {
            Some(handle_ptr) => {
                // SAFETY: lookup table only stores pointers that
                // were just boxed inside a live `DecompressPrivate`
                // and whose owning cinfo has not yet had
                // `jpeg_destroy_decompress` /
                // `jpeg_finish_decompress` /
                // `jpeg_abort_decompress` run on it (those
                // tear-downs all call `coef_unregister_array`).
                unsafe { (*handle_ptr).inner.clone() }
            }
            None => match materialize_foreign_coef_arrays(c, handle) {
                Ok(coeffs) => coeffs,
                Err(msg) => {
                    priv_state.last_error = CString::new(msg).unwrap_or_default();
                    return false;
                }
            },
        }
    };
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
    // Match the requested output coding mode — the same compress
    // parameters that drive `run_encoder_and_flush` for the pixel-encode
    // path also gate the coefficient-encode path, so jpegtran flags like
    // `-progressive`, `-arithmetic`, and `-optimize` produce the right
    // SOF / entropy variant. Pass `restart_in_rows` through to the
    // progressive writers (Huffman + arithmetic both honor it) so
    // `-restart Nb` produces row-mode markers.
    let restart_rows: Option<u16> = if c.restart_in_rows > 0 {
        Some(c.restart_in_rows as u16)
    } else {
        None
    };
    // 12-bit transcode (e.g. `monkey12.jpg`) MUST go through the
    // optimised Huffman writer: the non-optimised path uses standard
    // Annex K tables that only define DC categories 0..=11, so any
    // 12-bit DC diff that lands in category 12..=15 would silently
    // encode as a zero-bit code. Forcing the optimised path here means
    // a caller that copies critical parameters from a 12-bit source
    // (which sets `data_precision = 12` after `jpeg_set_defaults`
    // already left `optimize_coding = 0`) still produces a valid
    // stream. Mirrors the auto-promote in `jpeg_set_defaults` at the
    // top of this file.
    let force_optimize: bool = c.data_precision > 8;
    // Encode + push live in this scope so all heap-owning locals
    // (`adjusted`, `encoded`, `with_markers`) are dropped before any
    // `raise_cant_suspend` that may `longjmp` past Rust destructors.
    let push_status: Option<bool> = (|| -> Option<bool> {
        let bytes_result: libjpeg_turbo_rs::Result<Vec<u8>> =
            if c.progressive_mode != 0 && c.arith_code != 0 {
                libjpeg_turbo_rs::write_coefficients_progressive_arithmetic(&adjusted, restart_rows)
            } else if c.progressive_mode != 0 {
                libjpeg_turbo_rs::write_coefficients_progressive(&adjusted, restart_rows)
            } else if c.arith_code != 0 {
                libjpeg_turbo_rs::write_coefficients_arithmetic(&adjusted)
            } else if c.optimize_coding != 0 || force_optimize {
                libjpeg_turbo_rs::write_coefficients_optimized(&adjusted)
            } else {
                libjpeg_turbo_rs::write_coefficients(&adjusted)
            };
        let raw_encoded: Vec<u8> = match bytes_result {
            Ok(b) => b,
            Err(e) => {
                priv_state.last_error =
                    CString::new(format!("jpeg_finish_compress: {e}")).unwrap_or_default();
                return None;
            }
        };
        // Adobe APP14 handling — see the original comment block above
        // the function: 4-component output strips JFIF and substitutes
        // Adobe APP14; 3-component source with Adobe APP14 keeps both.
        let with_app14: Vec<u8> = if adjusted.components.len() == 4 {
            let transform: u8 = adjusted.adobe_transform.unwrap_or({
                if c.jpeg_color_space == JCS_YCCK {
                    2
                } else {
                    0
                }
            });
            let r = swap_jfif_for_adobe_app14(&raw_encoded, transform);
            drop(raw_encoded);
            r
        } else if let Some(transform) = adjusted.adobe_transform {
            let r = inject_adobe_app14_after_jfif(&raw_encoded, transform);
            drop(raw_encoded);
            r
        } else {
            raw_encoded
        };
        let with_markers: Vec<u8> =
            if priv_state.pending_markers.is_empty() && priv_state.icc_profile.is_none() {
                with_app14
            } else {
                let r = inject_markers_after_soi(&with_app14, priv_state);
                drop(with_app14);
                r
            };
        let ok = push_bytes_through_dest_mgr(c, priv_state, &with_markers);
        drop(with_markers);
        Some(ok)
    })();
    drop(adjusted);
    priv_state.pending_coef_arrays = std::ptr::null();

    match push_status {
        None => false,
        Some(completed) => {
            let failed: bool = !completed || dest_flush_failed(priv_state);
            finish_dest_flush(c, priv_state, completed);
            !failed
        }
    }
}

/// `jpeg_resync_to_restart(cinfo, desired) -> boolean`.
///
/// Default libjpeg behavior: always return TRUE so corrupted streams
/// attempt to resume at the next restart marker. Matches the function
/// libjpeg installs as the default `resync_to_restart` callback.
#[no_mangle]
pub extern "C" fn jpeg_resync_to_restart(_cinfo: *mut c_void, _desired: c_int) -> CBoolean {
    crate::unwind_guard!(0, { 1 })
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
    crate::unwind_guard!((), {
        if input_row.is_null() || output_row.is_null() || num_blocks == 0 {
            return;
        }
        let samples: usize = (num_blocks as usize).saturating_mul(64);
        // SAFETY: caller-supplied buffers of exactly `samples` i16 entries.
        unsafe {
            std::ptr::copy_nonoverlapping(input_row, output_row, samples);
        }
    })
}

/// `jdiv_round_up(a, b) -> long`.
///
/// Internal libjpeg utility: ceiling-divide for non-negative integers.
/// Matches `jutils.c::jdiv_round_up`.
#[no_mangle]
pub extern "C" fn jdiv_round_up(a: c_long, b: c_long) -> c_long {
    crate::unwind_guard!(0, {
        if b == 0 {
            return 0;
        }
        (a + b - 1) / b
    })
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
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
            c.image_width = width;
            c.image_height = height;
            c.input_components = input_components;
            c.in_color_space = in_color_space;
        }
    })
}

/// Test helper: flip the `progressive_mode` field of a compress cinfo
/// without going through the full `jpeg_simple_progression` path. Used to
/// verify the `jpeg_write_coefficients` → `jpeg_finish_compress` flow
/// honors progressive output between the two calls.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_progressive(cinfo: *mut c_void, progressive: c_int) {
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
            c.progressive_mode = progressive;
        }
    })
}

/// Test helper: set `restart_in_rows` directly. Mirrors `jpegtran
/// -restart Nrows` so the coefficient writer's row-mode → byte-mode
/// conversion can be exercised without a full simple_progression call.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_restart_in_rows(cinfo: *mut c_void, rows: c_int) {
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
            c.restart_in_rows = rows;
        }
    })
}

/// Test helper: set the public `restart_interval` field directly, matching
/// consumers such as OpenCV that configure block-mode restarts through the
/// ordinary v8 cinfo layout.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_restart_interval(cinfo: *mut c_void, interval: c_uint) {
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
            c.restart_interval = interval;
        }
    })
}

/// Test helper: toggle `arith_code` directly. Mirrors `jpegtran
/// -arithmetic` so the coefficient transcode path's arithmetic dispatch
/// can be exercised in tests.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_arith_code(cinfo: *mut c_void, arith: c_int) {
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
            c.arith_code = arith as CBoolean;
        }
    })
}

/// Test helper: set the public optimized-Huffman flag without relying on a
/// private Rust struct layout in dlopen-based integration tests.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_optimize_coding(cinfo: *mut c_void, optimize: c_int) {
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
            c.optimize_coding = optimize as CBoolean;
        }
    })
}

/// Test helper: set the public input-smoothing field at the classic C-ABI
/// boundary exercised by the scanline encoder matrix.
#[no_mangle]
pub extern "C" fn jpeg_capi_test_set_smoothing_factor(cinfo: *mut c_void, smoothing: c_int) {
    crate::unwind_guard!((), {
        if let Some(c) = unsafe { cinfo_compress_mut(cinfo) } {
            c.smoothing_factor = smoothing;
        }
    })
}

#[no_mangle]
pub extern "C" fn jpeg_capi_test_get_compress_state(
    cinfo: *mut c_void,
    out_num_components: *mut c_int,
    out_jpeg_color_space: *mut c_int,
    out_in_color_space: *mut c_int,
) {
    crate::unwind_guard!((), {
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
    })
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

#[cfg(test)]
mod tables_only_tests {
    use super::build_tables_only_datastream;

    /// Walk JPEG markers in `bytes`, returning `(tq_indices, dc_th_indices, ac_th_indices)`
    /// — the set of `Tq` numbers seen across all DQT segments, plus the `Th` numbers
    /// seen across DHT entries split by class (`Tc=0` DC, `Tc=1` AC).
    fn collect_table_indices(bytes: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut tq: Vec<u8> = Vec::new();
        let mut dc_th: Vec<u8> = Vec::new();
        let mut ac_th: Vec<u8> = Vec::new();
        let mut i: usize = 0;
        while i + 1 < bytes.len() {
            assert_eq!(bytes[i], 0xFF, "expected marker prefix at {i}");
            let code: u8 = bytes[i + 1];
            if code == 0xD8 || code == 0xD9 {
                i += 2;
                continue;
            }
            assert!(i + 4 <= bytes.len(), "truncated marker length at {i}");
            let seg_len: usize = ((bytes[i + 2] as usize) << 8) | (bytes[i + 3] as usize);
            assert!(seg_len >= 2 && i + 2 + seg_len <= bytes.len());
            let body: &[u8] = &bytes[i + 4..i + 2 + seg_len];
            match code {
                0xDB => {
                    // DQT: each entry is `Pq:4 | Tq:4`, then 64 (or 128 if Pq=1) bytes.
                    let mut p: usize = 0;
                    while p < body.len() {
                        let pq_tq: u8 = body[p];
                        let pq: u8 = pq_tq >> 4;
                        let tq_idx: u8 = pq_tq & 0x0F;
                        tq.push(tq_idx);
                        let entry_len: usize = if pq == 0 { 65 } else { 129 };
                        p += entry_len;
                    }
                }
                0xC4 => {
                    // DHT: each entry is `Tc:4 | Th:4`, 16 length bytes, then ∑ length values.
                    let mut p: usize = 0;
                    while p < body.len() {
                        let tc_th: u8 = body[p];
                        let tc: u8 = tc_th >> 4;
                        let th_idx: u8 = tc_th & 0x0F;
                        if tc == 0 {
                            dc_th.push(th_idx);
                        } else {
                            ac_th.push(th_idx);
                        }
                        let total_codes: usize =
                            body[p + 1..p + 17].iter().map(|&b| b as usize).sum();
                        p += 17 + total_codes;
                    }
                }
                _ => {}
            }
            i += 2 + seg_len;
        }
        (tq, dc_th, ac_th)
    }

    /// Walks markers and returns whether any DAC (0xCC) marker is present.
    fn has_dac(bytes: &[u8]) -> bool {
        let mut i: usize = 0;
        while i + 1 < bytes.len() {
            if bytes[i] != 0xFF {
                return false;
            }
            let code: u8 = bytes[i + 1];
            if code == 0xD8 || code == 0xD9 {
                i += 2;
                continue;
            }
            if i + 4 > bytes.len() {
                return false;
            }
            if code == 0xCC {
                return true;
            }
            let seg_len: usize = ((bytes[i + 2] as usize) << 8) | (bytes[i + 3] as usize);
            i += 2 + seg_len;
        }
        false
    }

    /// Walks markers and returns whether any DHT (0xC4) marker is present.
    fn has_dht(bytes: &[u8]) -> bool {
        let mut i: usize = 0;
        while i + 1 < bytes.len() {
            if bytes[i] != 0xFF {
                return false;
            }
            let code: u8 = bytes[i + 1];
            if code == 0xD8 || code == 0xD9 {
                i += 2;
                continue;
            }
            if i + 4 > bytes.len() {
                return false;
            }
            if code == 0xC4 {
                return true;
            }
            let seg_len: usize = ((bytes[i + 2] as usize) << 8) | (bytes[i + 3] as usize);
            i += 2 + seg_len;
        }
        false
    }

    #[test]
    fn huffman_datastream_emits_both_quant_and_all_huffman_tables() {
        let bytes: Vec<u8> = build_tables_only_datastream(75, /*arith=*/ false);
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "stream must begin with SOI");
        assert_eq!(
            &bytes[bytes.len() - 2..],
            &[0xFF, 0xD9],
            "stream must end with EOI"
        );
        let (tq, dc_th, ac_th) = collect_table_indices(&bytes);
        assert!(tq.contains(&0), "DQT[Tq=0] (luma) missing: {tq:?}");
        assert!(tq.contains(&1), "DQT[Tq=1] (chroma) missing: {tq:?}");
        assert!(dc_th.contains(&0), "DHT[Tc=0,Th=0] (DC luma) missing");
        assert!(dc_th.contains(&1), "DHT[Tc=0,Th=1] (DC chroma) missing");
        assert!(ac_th.contains(&0), "DHT[Tc=1,Th=0] (AC luma) missing");
        assert!(ac_th.contains(&1), "DHT[Tc=1,Th=1] (AC chroma) missing");
        assert!(
            !has_dac(&bytes),
            "Huffman tables-only stream must not contain DAC"
        );
    }

    #[test]
    fn arithmetic_datastream_omits_dht_and_dac() {
        // Upstream `jcmarker.c::write_tables_only` for arithmetic mode
        // emits SOI + DQT + EOI only. DHT is skipped (not the entropy
        // mode) and DAC is *not* emitted here either — DAC accompanies
        // the scan header in the abbreviated body stream, not this
        // tables-only datastream.
        let bytes: Vec<u8> = build_tables_only_datastream(75, /*arith=*/ true);
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "stream must begin with SOI");
        assert_eq!(
            &bytes[bytes.len() - 2..],
            &[0xFF, 0xD9],
            "stream must end with EOI"
        );
        let (tq, _, _) = collect_table_indices(&bytes);
        assert!(tq.contains(&0), "DQT[Tq=0] (luma) missing: {tq:?}");
        assert!(tq.contains(&1), "DQT[Tq=1] (chroma) missing: {tq:?}");
        assert!(
            !has_dht(&bytes),
            "arithmetic tables-only stream must not contain DHT"
        );
        assert!(
            !has_dac(&bytes),
            "arithmetic tables-only stream must not contain DAC \
             (DAC is emitted later with the scan header, not here)"
        );
    }

    #[test]
    fn datastream_well_formed_at_quality_extremes() {
        for q in [1u8, 50, 100] {
            for arith in [false, true] {
                let bytes: Vec<u8> = build_tables_only_datastream(q, arith);
                assert_eq!(
                    &bytes[..2],
                    &[0xFF, 0xD8],
                    "SOI missing at quality {q}, arith={arith}"
                );
                assert_eq!(
                    &bytes[bytes.len() - 2..],
                    &[0xFF, 0xD9],
                    "EOI missing at quality {q}, arith={arith}"
                );
                let (tq, _, _) = collect_table_indices(&bytes);
                assert!(
                    tq.contains(&0) && tq.contains(&1),
                    "missing quant tables at quality {q}, arith={arith}: {tq:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod marker_scan_tests {
    use super::{find_first_sos, scan_next_boundary, MarkerBoundary};

    /// Build a length-bearing marker segment: `FF <code> <len:2> <payload>`,
    /// where the 2-byte length counts itself + payload.
    fn seg(code: u8, payload: &[u8]) -> Vec<u8> {
        let len: usize = payload.len() + 2;
        let mut v: Vec<u8> = vec![0xFF, code, (len >> 8) as u8, (len & 0xFF) as u8];
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn first_sos_on_minimal_baseline() {
        let mut j: Vec<u8> = vec![0xFF, 0xD8]; // SOI
        j.extend(seg(0xDB, &[0u8; 65])); // DQT
        j.extend(seg(0xC0, &[0u8; 15])); // SOF0
        j.extend(seg(0xC4, &[0u8; 20])); // DHT
        j.extend(seg(0xDA, &[0u8; 8])); // SOS header
        let entropy_start: usize = j.len();
        // Entropy with a stuffed FF (FF 00) then EOI.
        j.extend_from_slice(&[0x12, 0x34, 0xFF, 0x00, 0x56]);
        j.extend_from_slice(&[0xFF, 0xD9]); // EOI
        assert_eq!(find_first_sos(&j), Some(entropy_start));
        // From the entropy start, the next boundary is EOI at end-of-buffer.
        assert_eq!(
            scan_next_boundary(&j, entropy_start),
            MarkerBoundary::Eoi(j.len())
        );
    }

    #[test]
    fn boundary_skips_restart_and_stuffing() {
        let mut j: Vec<u8> = vec![0xFF, 0xD8];
        j.extend(seg(0xDA, &[0u8; 8]));
        let entropy_start: usize = j.len();
        // entropy: literal data, stuffed FF, restart marker D2, more data, EOI
        j.extend_from_slice(&[0xAB, 0xFF, 0x00, 0xCD, 0xFF, 0xD2, 0xEF]);
        j.extend_from_slice(&[0xFF, 0xD9]);
        assert_eq!(
            scan_next_boundary(&j, entropy_start),
            MarkerBoundary::Eoi(j.len())
        );
    }

    #[test]
    fn two_scan_progressive_walks_each_boundary() {
        let mut j: Vec<u8> = vec![0xFF, 0xD8];
        j.extend(seg(0xC2, &[0u8; 15])); // SOF2 (progressive)
        j.extend(seg(0xC4, &[0u8; 20])); // DHT
        j.extend(seg(0xDA, &[0u8; 8])); // SOS #1
        let e1: usize = j.len();
        j.extend_from_slice(&[0x11, 0x22, 0xFF, 0x00]); // scan-1 entropy
                                                        // Inter-scan DHT, then SOS #2.
        j.extend(seg(0xC4, &[0u8; 20]));
        j.extend(seg(0xDA, &[0u8; 8])); // SOS #2
        let e2: usize = j.len();
        j.extend_from_slice(&[0x33, 0x44]); // scan-2 entropy
        j.extend_from_slice(&[0xFF, 0xD9]); // EOI
        assert_eq!(find_first_sos(&j), Some(e1));
        // From scan-1 entropy, skip the inter-scan DHT and reach SOS #2.
        assert_eq!(scan_next_boundary(&j, e1), MarkerBoundary::Sos(e2));
        // From scan-2 entropy, reach EOI.
        assert_eq!(scan_next_boundary(&j, e2), MarkerBoundary::Eoi(j.len()));
    }

    #[test]
    fn boundary_skips_tem_marker() {
        // A TEM marker (FF 01) is parameterless; the scanner must skip it like a
        // restart marker rather than reading the next bytes as a bogus length.
        let mut j: Vec<u8> = vec![0xFF, 0xD8];
        j.extend(seg(0xDA, &[0u8; 8]));
        let entropy_start: usize = j.len();
        j.extend_from_slice(&[0xAB, 0xFF, 0x01, 0xCD]); // entropy with a TEM
        j.extend_from_slice(&[0xFF, 0xD9]);
        assert_eq!(
            scan_next_boundary(&j, entropy_start),
            MarkerBoundary::Eoi(j.len())
        );
    }

    #[test]
    fn tables_only_has_no_sos() {
        let mut j: Vec<u8> = vec![0xFF, 0xD8];
        j.extend(seg(0xDB, &[0u8; 65])); // DQT
        j.extend(seg(0xC4, &[0u8; 20])); // DHT
        j.extend_from_slice(&[0xFF, 0xD9]); // EOI, no SOS
        assert_eq!(find_first_sos(&j), None);
    }

    #[test]
    fn truncated_sos_header_needs_more() {
        let mut j: Vec<u8> = vec![0xFF, 0xD8];
        j.extend(seg(0xC0, &[0u8; 15]));
        // SOS marker claiming an 8-byte segment but only 3 bytes present.
        j.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01]);
        assert_eq!(find_first_sos(&j), None);
    }

    #[test]
    fn unterminated_entropy_needs_more() {
        let mut j: Vec<u8> = vec![0xFF, 0xD8];
        j.extend(seg(0xDA, &[0u8; 8]));
        let entropy_start: usize = j.len();
        j.extend_from_slice(&[0x12, 0x34, 0x56]); // no marker terminates it
        assert!(matches!(
            scan_next_boundary(&j, entropy_start),
            MarkerBoundary::NeedMore(_)
        ));
    }

    #[test]
    fn needmore_resume_offset_avoids_rescan() {
        // The resume offset returned with NeedMore must let a caller continue
        // scanning newly-appended bytes without rescanning from the start.
        let mut j: Vec<u8> = vec![0xFF, 0xD8];
        j.extend(seg(0xDA, &[0u8; 8]));
        let entropy_start: usize = j.len();
        j.extend_from_slice(&[0x11, 0x22, 0x33]); // entropy, no terminator yet
        let resume: usize = match scan_next_boundary(&j, entropy_start) {
            MarkerBoundary::NeedMore(r) => r,
            other => panic!("expected NeedMore, got {other:?}"),
        };
        // Resume must not have rewound before the bytes already scanned.
        assert!(resume >= entropy_start && resume <= j.len());
        // Append an EOI; scanning from the resume offset must still find it.
        j.extend_from_slice(&[0xFF, 0xD9]);
        assert_eq!(scan_next_boundary(&j, resume), MarkerBoundary::Eoi(j.len()));
    }

    #[test]
    fn not_a_jpeg_returns_none() {
        assert_eq!(find_first_sos(&[0x00, 0x01, 0x02]), None);
        assert_eq!(find_first_sos(&[0xFF]), None);
        assert_eq!(find_first_sos(&[]), None);
    }
}
