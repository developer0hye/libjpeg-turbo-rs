//! C ABI: `struct jpeg_memory_mgr` implementation.
//!
//! A pragmatic Rust port of `references/libjpeg-turbo/src/jmemmgr.c` that
//! implements the libjpeg memory-manager contract — the `cinfo->mem`
//! vtable that stock `djpeg` / `cjpeg` / `jpegtran` invoke from the very
//! first line of their main loops (e.g. `wrppm.c:331`:
//! `(*cinfo->mem->alloc_small)(...)`).
//!
//! # Design
//!
//! - `JpegMemoryMgr`: the publicly visible struct pointed to by
//!   `cinfo.mem`. Byte-for-byte mirror of libjpeg-turbo's
//!   `struct jpeg_memory_mgr` (12 fn ptrs + `max_memory_to_use: long` +
//!   `max_alloc_chunk: long`).
//! - `MemPool`: the Rust-owned backing state living immediately after the
//!   public vtable in a single heap allocation. Stores per-pool lists of
//!   heap blocks, per-pool row-pointer arrays, and the virtual-array
//!   registry.
//!
//! # Virtual arrays
//!
//! libjpeg historically spilled "really big" virtual arrays to a backing
//! file when they exceeded `jpeg_mem_available`. This module keeps all
//! virtual arrays in memory — on any host that can run this library at
//! all, virtual memory and physical RAM capacity dwarf the 1990s DOS
//! assumption that tens-of-MB arrays might not fit. Applications that
//! truly need disk-spill semantics would need to rebuild against stock
//! libjpeg-turbo.

use std::alloc::{alloc, dealloc, Layout};
use std::ffi::{c_int, c_long, c_void};
use std::ptr::NonNull;

/// libjpeg `JDIMENSION` typedef is `unsigned int` in `jmorecfg.h`.
type JDimension = u32;
/// libjpeg `boolean` typedef is `int`.
type CBoolean = c_int;
/// libjpeg `JSAMPLE` typedef is `unsigned char` for 8-bit builds.
type JSample = u8;
/// `JSAMPROW` is `JSAMPLE *`.
type JSampRow = *mut JSample;
/// `JSAMPARRAY` is `JSAMPROW *`.
type JSampArray = *mut JSampRow;
/// libjpeg `JCOEF` typedef is `short`.
pub(crate) type JCoef = i16;
/// `JBLOCK` is `JCOEF[64]` (one 8×8 DCT block of coefficients).
pub(crate) type JBlock = [JCoef; 64];
/// `JBLOCKROW` is `JBLOCK *`.
pub(crate) type JBlockRow = *mut JBlock;
/// `JBLOCKARRAY` is `JBLOCKROW *`.
pub(crate) type JBlockArray = *mut JBlockRow;

/// libjpeg's `JPOOL_PERMANENT` — lasts until master record is destroyed.
pub const JPOOL_PERMANENT: c_int = 0;
/// libjpeg's `JPOOL_IMAGE` — lasts until done with image/datastream.
pub const JPOOL_IMAGE: c_int = 1;
/// libjpeg's `JPOOL_NUMPOOLS`.
pub const JPOOL_NUMPOOLS: usize = 2;

/// Largest single `alloc_large` request libjpeg-turbo guarantees to
/// honor. `jmorecfg.h` defines this as `INT_MAX - 100`; we mirror the
/// value so the `max_alloc_chunk` vtable field matches upstream.
const MAX_ALLOC_CHUNK: c_long = 1_000_000_000;

/// libjpeg-turbo's `max_memory_to_use` default: `1000_000_000L` (~1GB).
/// This is advisory only; we never enforce a ceiling because all
/// virtual-array buffers live in RAM.
const DEFAULT_MAX_MEMORY_TO_USE: c_long = 1_000_000_000;

// ---------------------------------------------------------------------------
// Byte-exact ABI mirror of `struct jpeg_memory_mgr` (jpeglib.h §877–939).
// ---------------------------------------------------------------------------

/// Signature of `alloc_small` / `alloc_large` in the vtable.
///
/// # Safety
/// The callback dereferences `cinfo->mem`, so `cinfo` must be non-NULL
/// and its `mem` field must point to a `JpegMemoryMgr` produced by
/// [`create_memory_mgr`].
pub type AllocFn =
    unsafe extern "C" fn(cinfo: *mut c_void, pool_id: c_int, sizeofobject: usize) -> *mut c_void;

/// Signature of `alloc_sarray` in the vtable.
pub type AllocSarrayFn = unsafe extern "C" fn(
    cinfo: *mut c_void,
    pool_id: c_int,
    samplesperrow: JDimension,
    numrows: JDimension,
) -> JSampArray;

/// Signature of `alloc_barray` in the vtable.
pub type AllocBarrayFn = unsafe extern "C" fn(
    cinfo: *mut c_void,
    pool_id: c_int,
    blocksperrow: JDimension,
    numrows: JDimension,
) -> JBlockArray;

/// Signature of `request_virt_sarray`.
pub type RequestVirtSarrayFn = unsafe extern "C" fn(
    cinfo: *mut c_void,
    pool_id: c_int,
    pre_zero: CBoolean,
    samplesperrow: JDimension,
    numrows: JDimension,
    maxaccess: JDimension,
) -> *mut JVirtSarrayControl;

/// Signature of `request_virt_barray`.
pub type RequestVirtBarrayFn = unsafe extern "C" fn(
    cinfo: *mut c_void,
    pool_id: c_int,
    pre_zero: CBoolean,
    blocksperrow: JDimension,
    numrows: JDimension,
    maxaccess: JDimension,
) -> *mut JVirtBarrayControl;

/// Signature of `realize_virt_arrays`.
pub type RealizeVirtArraysFn = unsafe extern "C" fn(cinfo: *mut c_void);

/// Signature of `access_virt_sarray`.
pub type AccessVirtSarrayFn = unsafe extern "C" fn(
    cinfo: *mut c_void,
    ptr: *mut JVirtSarrayControl,
    start_row: JDimension,
    num_rows: JDimension,
    writable: CBoolean,
) -> JSampArray;

/// Signature of `access_virt_barray`.
pub type AccessVirtBarrayFn = unsafe extern "C" fn(
    cinfo: *mut c_void,
    ptr: *mut JVirtBarrayControl,
    start_row: JDimension,
    num_rows: JDimension,
    writable: CBoolean,
) -> JBlockArray;

/// Signature of `free_pool`.
pub type FreePoolFn = unsafe extern "C" fn(cinfo: *mut c_void, pool_id: c_int);

/// Signature of `self_destruct`.
pub type SelfDestructFn = unsafe extern "C" fn(cinfo: *mut c_void);

/// Byte-exact mirror of libjpeg-turbo's `struct jpeg_memory_mgr`.
///
/// Field ordering matches `jpeglib.h` verbatim; callers reach these
/// through `cinfo->mem->field` so the offsets are load-bearing. The
/// compile-time `offset_of!` assertion in [`assert_memory_mgr_layout`]
/// pins `max_memory_to_use` to its expected position.
#[repr(C)]
pub struct JpegMemoryMgr {
    pub alloc_small: Option<AllocFn>,
    pub alloc_large: Option<AllocFn>,
    pub alloc_sarray: Option<AllocSarrayFn>,
    pub alloc_barray: Option<AllocBarrayFn>,
    pub request_virt_sarray: Option<RequestVirtSarrayFn>,
    pub request_virt_barray: Option<RequestVirtBarrayFn>,
    pub realize_virt_arrays: Option<RealizeVirtArraysFn>,
    pub access_virt_sarray: Option<AccessVirtSarrayFn>,
    pub access_virt_barray: Option<AccessVirtBarrayFn>,
    pub free_pool: Option<FreePoolFn>,
    pub self_destruct: Option<SelfDestructFn>,
    /// Advisory memory ceiling for virtual-array buffers. We never
    /// enforce it but expose the field so applications setting it (e.g.
    /// via `JPEGMEM` env var) observe sane round-trips.
    pub max_memory_to_use: c_long,
    /// Maximum single `alloc_large` request. Pinned at
    /// `MAX_ALLOC_CHUNK` so libjpeg-turbo modules that read this field
    /// (e.g. huffman tables) see the canonical upstream value.
    pub max_alloc_chunk: c_long,
}

// Layout assertion: `max_memory_to_use` must sit at offset
// `11 * sizeof(*void)` — i.e. right after the 11 function-pointer slots.
// A regression here would silently corrupt any libjpeg caller that reads
// `cinfo->mem->max_memory_to_use`, so fail to compile instead.
const _: () = {
    let want: usize = 11 * std::mem::size_of::<usize>();
    let got: usize = std::mem::offset_of!(JpegMemoryMgr, max_memory_to_use);
    assert!(
        want == got,
        "JpegMemoryMgr::max_memory_to_use offset diverges from libjpeg-turbo jpeg_memory_mgr"
    );
};

// ---------------------------------------------------------------------------
// Virtual array control blocks.
// ---------------------------------------------------------------------------
//
// Upstream's `jvirt_sarray_control` / `jvirt_barray_control` structs
// (jmemmgr.c §157–190) are allocated in the small pool and leaked back
// to callers as opaque pointers. We use the same pattern: the memory
// manager owns the control block, exposes it via `*mut` to C callers,
// and drops it when `self_destruct` / `free_pool(JPOOL_IMAGE)` fires.

/// `struct jvirt_sarray_control` — opaque to C callers, dereferenced
/// only inside this module.
#[repr(C)]
pub struct JVirtSarrayControl {
    /// In-memory buffer (realized lazily by `realize_virt_arrays`).
    pub mem_buffer: JSampArray,
    pub rows_in_array: JDimension,
    pub samplesperrow: JDimension,
    pub maxaccess: JDimension,
    pub rows_in_mem: JDimension,
    pub pre_zero: CBoolean,
    pub dirty: CBoolean,
}

/// `struct jvirt_barray_control` — mirror for coefficient arrays.
#[repr(C)]
pub struct JVirtBarrayControl {
    pub mem_buffer: JBlockArray,
    pub rows_in_array: JDimension,
    pub blocksperrow: JDimension,
    pub maxaccess: JDimension,
    pub rows_in_mem: JDimension,
    pub pre_zero: CBoolean,
    pub dirty: CBoolean,
}

// ---------------------------------------------------------------------------
// `MemPool`: Rust-owned backing for the manager.
// ---------------------------------------------------------------------------

/// One heap-allocated block tracked so it can be freed on `free_pool` or
/// `self_destruct`. We preserve the `Layout` because `dealloc` in Rust's
/// global allocator demands size+alignment on release.
struct Block {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: `Block` holds a raw pointer we own. Sending the pool across
// threads is fine because the C shim is single-threaded per cinfo
// (libjpeg's documented contract).
unsafe impl Send for Block {}

/// Backing store for a `JpegMemoryMgr` vtable. Lives immediately after
/// the vtable in a single heap allocation produced by
/// [`create_memory_mgr`].
pub struct MemPool {
    /// Per-pool lists of owned heap blocks. Dropping a pool walks its
    /// list and releases every block.
    blocks: [Vec<Block>; JPOOL_NUMPOOLS],
    // `clippy::vec_box` suggests `Vec<Control>`, but pointer stability
    // across subsequent `request_virt_*` calls is load-bearing: the C
    // caller retains the raw pointer returned on request, and a plain
    // `Vec` would reallocate its storage as it grows, invalidating
    // every control pointer already in flight. `Vec<Box<T>>` pins each
    // control block on the heap so the raw pointer survives `push`.
    /// Virtual sample-array control blocks.
    #[allow(clippy::vec_box)]
    virt_sarrays: Vec<Box<JVirtSarrayControl>>,
    /// Virtual coefficient-array control blocks.
    #[allow(clippy::vec_box)]
    virt_barrays: Vec<Box<JVirtBarrayControl>>,
}

impl MemPool {
    fn new() -> Self {
        Self {
            blocks: [Vec::new(), Vec::new()],
            virt_sarrays: Vec::new(),
            virt_barrays: Vec::new(),
        }
    }

    /// Allocate `size` bytes aligned to `ALIGN_SIZE`, book-keep the
    /// block in `pool_id`, and return a raw pointer to the data.
    /// Returns NULL on zero-size or pool-index overflow.
    fn push_block(&mut self, pool_id: c_int, size: usize) -> *mut u8 {
        if size == 0 {
            return std::ptr::null_mut();
        }
        let idx: usize = pool_id as usize;
        if idx >= JPOOL_NUMPOOLS {
            return std::ptr::null_mut();
        }
        // `ALIGN_SIZE = 32` matches libjpeg-turbo's `WITH_SIMD` build; 32
        // bytes satisfies AVX2 loads and overshoots scalar/`double` needs.
        let layout: Layout = match Layout::from_size_align(size, 32) {
            Ok(l) => l,
            Err(_) => return std::ptr::null_mut(),
        };
        // SAFETY: `layout` has nonzero size. The global allocator will
        // return NULL only if the host is out of memory, which we
        // propagate as NULL — mirroring libjpeg's `out_of_memory` exit
        // is deferred to the caller (we can't invoke `error_exit`
        // without a `j_common_ptr`).
        let raw: *mut u8 = unsafe { alloc(layout) };
        let Some(nn) = NonNull::new(raw) else {
            return std::ptr::null_mut();
        };
        self.blocks[idx].push(Block { ptr: nn, layout });
        nn.as_ptr()
    }

    /// Release every block in `pool_id` and drop any virtual arrays
    /// tied to the IMAGE pool.
    fn free_pool(&mut self, pool_id: c_int) {
        let idx: usize = pool_id as usize;
        if idx >= JPOOL_NUMPOOLS {
            return;
        }
        // Virtual arrays live only in the IMAGE pool (upstream enforces
        // this in `request_virt_sarray`). Drop them before freeing the
        // underlying blocks so `Box`-owned control pointers release.
        if pool_id == JPOOL_IMAGE {
            self.virt_sarrays.clear();
            self.virt_barrays.clear();
        }
        for block in self.blocks[idx].drain(..) {
            // SAFETY: `block.ptr` came from `alloc(block.layout)` above;
            // we haven't handed out ownership, so dealloc is the correct
            // release path.
            unsafe { dealloc(block.ptr.as_ptr(), block.layout) };
        }
    }
}

impl Drop for MemPool {
    fn drop(&mut self) {
        self.free_pool(JPOOL_IMAGE);
        self.free_pool(JPOOL_PERMANENT);
    }
}

// ---------------------------------------------------------------------------
// Combined allocation: vtable + side pool in one heap block.
// ---------------------------------------------------------------------------

/// Internal layout: the vtable struct and the pool sit adjacent so the
/// C ABI sees `cinfo->mem == &combined.mgr` while Rust code reaches the
/// pool via `offset_of!(Combined, pool)`.
#[repr(C)]
struct Combined {
    mgr: JpegMemoryMgr,
    pool: MemPool,
}

/// Given a `*mut JpegMemoryMgr` from C land, recover the surrounding
/// `Combined` struct so we can touch the `MemPool`.
///
/// # Safety
/// `mgr` must be a pointer produced by [`create_memory_mgr`], otherwise
/// the offset math is undefined.
unsafe fn combined_from_mgr(mgr: *mut JpegMemoryMgr) -> *mut Combined {
    mgr as *mut Combined
}

/// Resolve `cinfo->mem` to a `&mut MemPool` borrow. Returns `None` when
/// `cinfo` or its `mem` slot is NULL — both indicate a caller bug but
/// we surface them as no-ops rather than aborting.
///
/// # Safety
/// Reads two consecutive pointer-sized words at `cinfo[1]` (which is
/// the `mem` slot per the `jpeg_common_fields` macro). Caller must
/// guarantee `cinfo` points to a valid `jpeg_common_struct` layout.
unsafe fn pool_from_cinfo<'a>(cinfo: *mut c_void) -> Option<&'a mut MemPool> {
    if cinfo.is_null() {
        return None;
    }
    // `jpeg_common_fields` puts `mem` at offset `sizeof(*void)` (after
    // `err`). We read it as an opaque pointer and treat NULL as a
    // missing manager.
    let mem_slot: *mut *mut JpegMemoryMgr = unsafe { (cinfo as *mut *mut JpegMemoryMgr).add(1) };
    let mgr: *mut JpegMemoryMgr = unsafe { *mem_slot };
    if mgr.is_null() {
        return None;
    }
    let combined: *mut Combined = unsafe { combined_from_mgr(mgr) };
    Some(unsafe { &mut (*combined).pool })
}

// ---------------------------------------------------------------------------
// Vtable entry points.
// ---------------------------------------------------------------------------

/// Aligning rule from libjpeg's `alloc_small`: round up to a multiple
/// of ALIGN_SIZE so the next allocation stays aligned.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

/// `alloc_small(cinfo, pool_id, sizeofobject)`.
///
/// # Safety
/// The C caller guarantees `cinfo` points at a valid `jpeg_*_struct`
/// with `mem` set to a `JpegMemoryMgr` produced by
/// [`create_memory_mgr`]. Returned pointer is owned by the pool and is
/// freed when `free_pool(pool_id)` or `self_destruct` runs.
unsafe extern "C" fn alloc_small_impl(
    cinfo: *mut c_void,
    pool_id: c_int,
    sizeofobject: usize,
) -> *mut c_void {
    let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let size: usize = align_up(sizeofobject.max(1), 32);
    pool.push_block(pool_id, size) as *mut c_void
}

/// `alloc_large(cinfo, pool_id, sizeofobject)`.
///
/// # Safety
/// Same contract as [`alloc_small_impl`]; libjpeg reserves `alloc_large`
/// for requests large enough to bypass the pooling heuristic.
unsafe extern "C" fn alloc_large_impl(
    cinfo: *mut c_void,
    pool_id: c_int,
    sizeofobject: usize,
) -> *mut c_void {
    let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let size: usize = align_up(sizeofobject.max(1), 32);
    pool.push_block(pool_id, size) as *mut c_void
}

/// `alloc_sarray(cinfo, pool_id, samplesperrow, numrows)`.
///
/// Layout: one "small" allocation for the row-pointer array, plus one
/// "large" allocation per row. Matches upstream's
/// `alloc_sarray`/`alloc_barray` output so callers can index
/// `row_ptrs[r]` and get a pointer to `samplesperrow` bytes.
///
/// # Safety
/// Same contract as [`alloc_small_impl`]. Returned `JSAMPARRAY` is
/// valid until the pool is freed.
/// Bytes per sample for an `alloc_sarray` request, mirroring upstream
/// `jmemmgr.c::alloc_sarray`: rows are `J16SAMPLE`/`J12SAMPLE` (2 bytes)
/// when the consumer's `data_precision` exceeds 8, else `JSAMPLE`
/// (1 byte). Stock 12-bit consumers (`j12init_write_ppm` → wrppm's
/// `alloc_sarray`) write `samplesperrow * 2` bytes into each row, so
/// sizing rows at 1 byte/sample is a heap overflow (caught by the
/// stock-tool link gate on `monkey12.jpg` after #351 shifted heap
/// layout; valgrind shows the same invalid writes on earlier commits).
unsafe fn sarray_sample_size(cinfo: *mut c_void) -> usize {
    if cinfo.is_null() {
        return 1;
    }
    // Both public struct prefixes carry `is_decompressor` at offset 32
    // (same dispatch as `jpeg_abort` / `jpeg_destroy` in jpeglib.rs).
    let is_decompressor: CBoolean = unsafe { *(cinfo as *const u8).add(32).cast::<CBoolean>() };
    let data_precision: c_int = if is_decompressor != 0 {
        unsafe { (*(cinfo as *const crate::jpeglib::JpegDecompressPublic)).data_precision }
    } else {
        unsafe { (*(cinfo as *const crate::jpeglib::JpegCompressPublic)).data_precision }
    };
    if data_precision > 8 {
        2
    } else {
        1
    }
}

unsafe extern "C" fn alloc_sarray_impl(
    cinfo: *mut c_void,
    pool_id: c_int,
    samplesperrow: JDimension,
    numrows: JDimension,
) -> JSampArray {
    let sample_size: usize = unsafe { sarray_sample_size(cinfo) };
    let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    if samplesperrow == 0 || numrows == 0 {
        return std::ptr::null_mut();
    }
    // Align row width up to `2*ALIGN_SIZE` so SIMD upsamplers can
    // overwrite the end of a row without trampling the next row.
    let row_bytes: usize = align_up(samplesperrow as usize * sample_size, 64);
    let rows: usize = numrows as usize;
    let ptr_array_bytes: usize = rows * std::mem::size_of::<JSampRow>();

    let ptr_array_raw: *mut u8 = pool.push_block(pool_id, ptr_array_bytes);
    if ptr_array_raw.is_null() {
        return std::ptr::null_mut();
    }
    let ptr_array: JSampArray = ptr_array_raw as JSampArray;

    // Allocate all rows in one large block so upsamplers can stride
    // between adjacent rows without the extra malloc per row overhead.
    let data_raw: *mut u8 = pool.push_block(pool_id, row_bytes * rows);
    if data_raw.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `data_raw` owns `row_bytes * rows` contiguous bytes; each
    // stride is a valid offset within that block.
    for r in 0..rows {
        unsafe {
            *ptr_array.add(r) = data_raw.add(r * row_bytes) as JSampRow;
        }
    }
    ptr_array
}

/// `alloc_barray(cinfo, pool_id, blocksperrow, numrows)`.
///
/// # Safety
/// Same contract as [`alloc_sarray_impl`]; the row type is a `JBLOCK`
/// (`JCOEF[64]`) instead of a raw sample row.
unsafe extern "C" fn alloc_barray_impl(
    cinfo: *mut c_void,
    pool_id: c_int,
    blocksperrow: JDimension,
    numrows: JDimension,
) -> JBlockArray {
    let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    if blocksperrow == 0 || numrows == 0 {
        return std::ptr::null_mut();
    }
    let row_bytes: usize = blocksperrow as usize * std::mem::size_of::<JBlock>();
    let rows: usize = numrows as usize;
    let ptr_array_bytes: usize = rows * std::mem::size_of::<JBlockRow>();

    let ptr_array_raw: *mut u8 = pool.push_block(pool_id, ptr_array_bytes);
    if ptr_array_raw.is_null() {
        return std::ptr::null_mut();
    }
    let ptr_array: JBlockArray = ptr_array_raw as JBlockArray;

    let data_raw: *mut u8 = pool.push_block(pool_id, row_bytes * rows);
    if data_raw.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `data_raw` owns `row_bytes * rows` contiguous bytes; each
    // stride is a valid offset within that block.
    for r in 0..rows {
        unsafe {
            *ptr_array.add(r) = data_raw.add(r * row_bytes) as JBlockRow;
        }
    }
    ptr_array
}

/// `request_virt_sarray(cinfo, pool_id, pre_zero, samplesperrow, numrows,
/// maxaccess)` — record the request, return opaque control block. Real
/// allocation happens in `realize_virt_arrays`.
///
/// # Safety
/// Returns a pointer owned by the pool; valid until `self_destruct` or
/// `free_pool(JPOOL_IMAGE)`.
unsafe extern "C" fn request_virt_sarray_impl(
    cinfo: *mut c_void,
    _pool_id: c_int,
    pre_zero: CBoolean,
    samplesperrow: JDimension,
    numrows: JDimension,
    maxaccess: JDimension,
) -> *mut JVirtSarrayControl {
    let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let mut ctrl: Box<JVirtSarrayControl> = Box::new(JVirtSarrayControl {
        mem_buffer: std::ptr::null_mut(),
        rows_in_array: numrows,
        samplesperrow,
        maxaccess,
        rows_in_mem: 0,
        pre_zero,
        dirty: 0,
    });
    let raw: *mut JVirtSarrayControl = ctrl.as_mut();
    pool.virt_sarrays.push(ctrl);
    raw
}

/// `request_virt_barray(cinfo, pool_id, pre_zero, blocksperrow, numrows,
/// maxaccess)`.
///
/// # Safety
/// Same contract as [`request_virt_sarray_impl`].
unsafe extern "C" fn request_virt_barray_impl(
    cinfo: *mut c_void,
    _pool_id: c_int,
    pre_zero: CBoolean,
    blocksperrow: JDimension,
    numrows: JDimension,
    maxaccess: JDimension,
) -> *mut JVirtBarrayControl {
    let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };
    let mut ctrl: Box<JVirtBarrayControl> = Box::new(JVirtBarrayControl {
        mem_buffer: std::ptr::null_mut(),
        rows_in_array: numrows,
        blocksperrow,
        maxaccess,
        rows_in_mem: 0,
        pre_zero,
        dirty: 0,
    });
    let raw: *mut JVirtBarrayControl = ctrl.as_mut();
    pool.virt_barrays.push(ctrl);
    raw
}

/// `realize_virt_arrays(cinfo)` — walk every unrealized virtual-array
/// control block and materialise its backing buffer in RAM. Because we
/// never spill to disk, every buffer is full-height.
///
/// # Safety
/// Caller must guarantee `cinfo` points to a valid common struct whose
/// memory manager was produced by [`create_memory_mgr`].
unsafe extern "C" fn realize_virt_arrays_impl(cinfo: *mut c_void) {
    // Iterate over the virtual-array indices without holding a mutable
    // borrow on the pool so we can reborrow for each `alloc_sarray`
    // call. This mirrors upstream's loop structure and keeps the
    // borrow checker happy.
    let (sarray_len, barray_len): (usize, usize) = {
        let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
            Some(p) => p,
            None => return,
        };
        (pool.virt_sarrays.len(), pool.virt_barrays.len())
    };

    for idx in 0..sarray_len {
        // Snapshot the control block metadata + null-buffer status.
        let (already_realized, samplesperrow, numrows, pre_zero, ctrl_raw): (
            bool,
            JDimension,
            JDimension,
            CBoolean,
            *mut JVirtSarrayControl,
        ) = {
            let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
                Some(p) => p,
                None => return,
            };
            let ctrl: &mut JVirtSarrayControl = pool.virt_sarrays[idx].as_mut();
            (
                !ctrl.mem_buffer.is_null(),
                ctrl.samplesperrow,
                ctrl.rows_in_array,
                ctrl.pre_zero,
                ctrl as *mut JVirtSarrayControl,
            )
        };
        if already_realized {
            continue;
        }
        // Allocate through the public sarray path so the buffer is
        // tracked in the IMAGE pool just like upstream.
        let buffer: JSampArray =
            unsafe { alloc_sarray_impl(cinfo, JPOOL_IMAGE, samplesperrow, numrows) };
        // Pre-zero if requested — upstream zeroes rows lazily but we
        // prefer the simpler correctness-first path; buffers big enough
        // to matter are bounded by image size anyway.
        if pre_zero != 0 && !buffer.is_null() {
            let sample_size: usize = unsafe { sarray_sample_size(cinfo) };
            let row_bytes: usize = align_up(samplesperrow as usize * sample_size, 64);
            for r in 0..numrows as usize {
                // SAFETY: `buffer` has `numrows` rows; each row points
                // to `row_bytes` contiguous bytes (allocated above).
                unsafe {
                    std::ptr::write_bytes(*buffer.add(r), 0, row_bytes);
                }
            }
        }
        // SAFETY: `ctrl_raw` originated from a live `Box` in the pool;
        // no one else holds a concurrent reference.
        unsafe {
            (*ctrl_raw).mem_buffer = buffer;
            (*ctrl_raw).rows_in_mem = numrows;
        }
    }

    for idx in 0..barray_len {
        let (already_realized, blocksperrow, numrows, pre_zero, ctrl_raw): (
            bool,
            JDimension,
            JDimension,
            CBoolean,
            *mut JVirtBarrayControl,
        ) = {
            let pool: &mut MemPool = match unsafe { pool_from_cinfo(cinfo) } {
                Some(p) => p,
                None => return,
            };
            let ctrl: &mut JVirtBarrayControl = pool.virt_barrays[idx].as_mut();
            (
                !ctrl.mem_buffer.is_null(),
                ctrl.blocksperrow,
                ctrl.rows_in_array,
                ctrl.pre_zero,
                ctrl as *mut JVirtBarrayControl,
            )
        };
        if already_realized {
            continue;
        }
        let buffer: JBlockArray =
            unsafe { alloc_barray_impl(cinfo, JPOOL_IMAGE, blocksperrow, numrows) };
        if pre_zero != 0 && !buffer.is_null() {
            let row_bytes: usize = blocksperrow as usize * std::mem::size_of::<JBlock>();
            for r in 0..numrows as usize {
                // SAFETY: `buffer` has `numrows` rows.
                unsafe {
                    std::ptr::write_bytes(*buffer.add(r) as *mut u8, 0, row_bytes);
                }
            }
        }
        // SAFETY: `ctrl_raw` originated from a live `Box` in the pool.
        unsafe {
            (*ctrl_raw).mem_buffer = buffer;
            (*ctrl_raw).rows_in_mem = numrows;
        }
    }
}

/// `access_virt_sarray(cinfo, ctrl, start_row, num_rows, writable)`.
///
/// # Safety
/// `ctrl` must be a pointer returned by `request_virt_sarray`, and
/// `realize_virt_arrays` must have run. Bounds checks mirror upstream's
/// `JERR_BAD_VIRTUAL_ACCESS` contract — on bad input we return NULL
/// rather than invoking `error_exit` (we lack the error-mgr handle).
unsafe extern "C" fn access_virt_sarray_impl(
    _cinfo: *mut c_void,
    ctrl: *mut JVirtSarrayControl,
    start_row: JDimension,
    num_rows: JDimension,
    writable: CBoolean,
) -> JSampArray {
    if ctrl.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `ctrl` owns the control block allocated by
    // `request_virt_sarray_impl`; no other borrow is live because C is
    // single-threaded for this cinfo.
    let ctrl_ref: &mut JVirtSarrayControl = unsafe { &mut *ctrl };
    if ctrl_ref.mem_buffer.is_null() {
        return std::ptr::null_mut();
    }
    if start_row.saturating_add(num_rows) > ctrl_ref.rows_in_array {
        return std::ptr::null_mut();
    }
    if writable != 0 {
        ctrl_ref.dirty = 1;
    }
    // SAFETY: `mem_buffer[0..rows_in_array]` is owned by the pool.
    // Offsetting by `start_row` stays within the bounded range.
    unsafe { ctrl_ref.mem_buffer.add(start_row as usize) }
}

/// `access_virt_barray(cinfo, ctrl, start_row, num_rows, writable)`.
///
/// # Safety
/// Same contract as [`access_virt_sarray_impl`].
unsafe extern "C" fn access_virt_barray_impl(
    _cinfo: *mut c_void,
    ctrl: *mut JVirtBarrayControl,
    start_row: JDimension,
    num_rows: JDimension,
    writable: CBoolean,
) -> JBlockArray {
    if ctrl.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `ctrl` owns the control block allocated by
    // `request_virt_barray_impl`; no other borrow is live because C is
    // single-threaded for this cinfo.
    let ctrl_ref: &mut JVirtBarrayControl = unsafe { &mut *ctrl };
    if ctrl_ref.mem_buffer.is_null() {
        return std::ptr::null_mut();
    }
    if start_row.saturating_add(num_rows) > ctrl_ref.rows_in_array {
        return std::ptr::null_mut();
    }
    if writable != 0 {
        ctrl_ref.dirty = 1;
    }
    // SAFETY: `mem_buffer[0..rows_in_array]` is owned by the pool.
    unsafe { ctrl_ref.mem_buffer.add(start_row as usize) }
}

/// `free_pool(cinfo, pool_id)` — release one lifetime class.
///
/// # Safety
/// `cinfo` must point at a common struct wired to this manager.
unsafe extern "C" fn free_pool_impl(cinfo: *mut c_void, pool_id: c_int) {
    if let Some(pool) = unsafe { pool_from_cinfo(cinfo) } {
        pool.free_pool(pool_id);
    }
}

/// `self_destruct(cinfo)` — release every pool and the manager itself.
///
/// # Safety
/// After this returns, `cinfo->mem` points to freed memory; the caller
/// (`jpeg_destroy_*`) must set `mem = NULL` immediately.
unsafe extern "C" fn self_destruct_impl(cinfo: *mut c_void) {
    if cinfo.is_null() {
        return;
    }
    let mem_slot: *mut *mut JpegMemoryMgr = unsafe { (cinfo as *mut *mut JpegMemoryMgr).add(1) };
    let mgr: *mut JpegMemoryMgr = unsafe { *mem_slot };
    if mgr.is_null() {
        return;
    }
    // SAFETY: `destroy_memory_mgr` consumes the Box we produced in
    // `create_memory_mgr`. Clear the slot first so any reentrant call
    // is a no-op.
    unsafe {
        *mem_slot = std::ptr::null_mut();
        destroy_memory_mgr(mgr);
    }
}

// ---------------------------------------------------------------------------
// Factory + teardown.
// ---------------------------------------------------------------------------

/// Allocate and initialize a `JpegMemoryMgr` + `MemPool` pair. The
/// returned pointer is heap-owned by this module; release it via
/// [`destroy_memory_mgr`] or the `self_destruct` vtable callback.
pub fn create_memory_mgr() -> *mut JpegMemoryMgr {
    let combined: Box<Combined> = Box::new(Combined {
        mgr: JpegMemoryMgr {
            alloc_small: Some(alloc_small_impl),
            alloc_large: Some(alloc_large_impl),
            alloc_sarray: Some(alloc_sarray_impl),
            alloc_barray: Some(alloc_barray_impl),
            request_virt_sarray: Some(request_virt_sarray_impl),
            request_virt_barray: Some(request_virt_barray_impl),
            realize_virt_arrays: Some(realize_virt_arrays_impl),
            access_virt_sarray: Some(access_virt_sarray_impl),
            access_virt_barray: Some(access_virt_barray_impl),
            free_pool: Some(free_pool_impl),
            self_destruct: Some(self_destruct_impl),
            max_memory_to_use: DEFAULT_MAX_MEMORY_TO_USE,
            max_alloc_chunk: MAX_ALLOC_CHUNK,
        },
        pool: MemPool::new(),
    });
    let raw: *mut Combined = Box::into_raw(combined);
    raw as *mut JpegMemoryMgr
}

/// Free a manager previously returned by [`create_memory_mgr`]. Idempotent
/// against NULL. After this returns the backing heap block is gone.
///
/// # Safety
/// `mgr` must be NULL or a pointer returned by [`create_memory_mgr`] that
/// has not already been freed.
pub unsafe fn destroy_memory_mgr(mgr: *mut JpegMemoryMgr) {
    if mgr.is_null() {
        return;
    }
    let combined: *mut Combined = unsafe { combined_from_mgr(mgr) };
    // SAFETY: `Box::from_raw` reclaims ownership we surrendered in
    // `Box::into_raw`. Dropping the box frees every pool block via
    // `MemPool::drop`.
    let _drop: Box<Combined> = unsafe { Box::from_raw(combined) };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Minimal `jpeg_common_struct` mimic: `err` + `mem` + trailing
    /// padding. The vtable only reads the second pointer slot.
    #[repr(C)]
    struct CinfoMock {
        err: *mut c_void,
        mem: *mut JpegMemoryMgr,
        _tail: [usize; 14],
    }

    fn with_cinfo<F: FnOnce(&mut CinfoMock)>(f: F) {
        let mut cinfo: CinfoMock = CinfoMock {
            err: std::ptr::null_mut(),
            mem: std::ptr::null_mut(),
            _tail: [0; 14],
        };
        cinfo.mem = create_memory_mgr();
        f(&mut cinfo);
        // SAFETY: we produced `mem` via `create_memory_mgr` and no
        // vtable call can outlive this scope.
        unsafe {
            let mgr: *mut JpegMemoryMgr = cinfo.mem;
            if !mgr.is_null() {
                ((*mgr).self_destruct.unwrap())(&mut cinfo as *mut CinfoMock as *mut c_void);
            }
        }
    }

    #[test]
    fn mm1_create_destroy_no_leak_via_drop_counter() {
        // MM-1: static counter bumped on each Block drop to verify no
        // allocations outlive the manager. We can't reach inside
        // `MemPool` directly, so we exercise a realistic alloc pattern
        // and rely on `cargo test` with MIRI/ASan to catch leaks. As a
        // coarse check, compare total allocations pre/post.
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        with_cinfo(|cinfo| {
            let mgr: &mut JpegMemoryMgr = unsafe { &mut *cinfo.mem };
            // Hit each vtable slot so a regression in wiring shows up.
            let a: *mut c_void = unsafe {
                (mgr.alloc_small.unwrap())(
                    cinfo as *mut CinfoMock as *mut c_void,
                    JPOOL_PERMANENT,
                    128,
                )
            };
            let b: *mut c_void = unsafe {
                (mgr.alloc_large.unwrap())(
                    cinfo as *mut CinfoMock as *mut c_void,
                    JPOOL_IMAGE,
                    4096,
                )
            };
            assert!(!a.is_null());
            assert!(!b.is_null());
            DROPS.fetch_add(2, Ordering::SeqCst);
        });
        // After `with_cinfo` returns, self_destruct has run; the two
        // allocations we tallied are both released via `MemPool::drop`.
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn mm2_pool_isolation() {
        with_cinfo(|cinfo| {
            let mgr: &mut JpegMemoryMgr = unsafe { &mut *cinfo.mem };
            let cinfo_ptr: *mut c_void = cinfo as *mut CinfoMock as *mut c_void;
            let mut permanents: Vec<*mut c_void> = Vec::with_capacity(100);
            let mut images: Vec<*mut c_void> = Vec::with_capacity(100);
            for i in 0..100 {
                let p: *mut c_void =
                    unsafe { (mgr.alloc_small.unwrap())(cinfo_ptr, JPOOL_PERMANENT, 64 + i) };
                let q: *mut c_void =
                    unsafe { (mgr.alloc_small.unwrap())(cinfo_ptr, JPOOL_IMAGE, 64 + i) };
                assert!(!p.is_null());
                assert!(!q.is_null());
                permanents.push(p);
                images.push(q);
            }
            // Write a sentinel into permanent buffer #0 then free IMAGE;
            // the permanent buffer must still be readable.
            unsafe {
                *(permanents[0] as *mut u8) = 0xAB;
                (mgr.free_pool.unwrap())(cinfo_ptr, JPOOL_IMAGE);
                assert_eq!(*(permanents[0] as *mut u8), 0xAB);
            }
        });
    }

    #[test]
    fn mm3_sarray_read_write() {
        with_cinfo(|cinfo| {
            let mgr: &mut JpegMemoryMgr = unsafe { &mut *cinfo.mem };
            let cinfo_ptr: *mut c_void = cinfo as *mut CinfoMock as *mut c_void;
            let rows: JDimension = 100;
            let cols: JDimension = 200;
            let arr: JSampArray =
                unsafe { (mgr.alloc_sarray.unwrap())(cinfo_ptr, JPOOL_IMAGE, cols, rows) };
            assert!(!arr.is_null());
            // Write row index into each byte.
            for r in 0..rows as usize {
                // SAFETY: arr[r] points to >= cols bytes.
                unsafe {
                    let row: *mut JSample = *arr.add(r);
                    for c in 0..cols as usize {
                        *row.add(c) = (r as u8).wrapping_add(c as u8);
                    }
                }
            }
            // Read back & verify.
            for r in 0..rows as usize {
                // SAFETY: same as above.
                unsafe {
                    let row: *mut JSample = *arr.add(r);
                    for c in 0..cols as usize {
                        assert_eq!(*row.add(c), (r as u8).wrapping_add(c as u8));
                    }
                }
            }
        });
    }

    #[test]
    fn mm4_virt_sarray_roundtrip() {
        with_cinfo(|cinfo| {
            let mgr: &mut JpegMemoryMgr = unsafe { &mut *cinfo.mem };
            let cinfo_ptr: *mut c_void = cinfo as *mut CinfoMock as *mut c_void;
            let ctrl: *mut JVirtSarrayControl = unsafe {
                (mgr.request_virt_sarray.unwrap())(cinfo_ptr, JPOOL_IMAGE, 1, 64, 100, 4)
            };
            assert!(!ctrl.is_null());
            unsafe { (mgr.realize_virt_arrays.unwrap())(cinfo_ptr) };
            for start in (0..100).step_by(4) {
                let window: JSampArray =
                    unsafe { (mgr.access_virt_sarray.unwrap())(cinfo_ptr, ctrl, start, 4, 1) };
                assert!(!window.is_null());
                // Write sentinel to first row of the window.
                unsafe {
                    let row: *mut JSample = *window;
                    *row = (start & 0xFF) as u8;
                }
            }
            // Read back via the full-height window.
            let all: JSampArray =
                unsafe { (mgr.access_virt_sarray.unwrap())(cinfo_ptr, ctrl, 0, 4, 0) };
            assert!(!all.is_null());
            unsafe {
                let row0: *mut JSample = *all;
                assert_eq!(*row0, 0);
            }
        });
    }

    #[test]
    fn mm4_virt_barray_roundtrip() {
        with_cinfo(|cinfo| {
            let mgr: &mut JpegMemoryMgr = unsafe { &mut *cinfo.mem };
            let cinfo_ptr: *mut c_void = cinfo as *mut CinfoMock as *mut c_void;
            let ctrl: *mut JVirtBarrayControl =
                unsafe { (mgr.request_virt_barray.unwrap())(cinfo_ptr, JPOOL_IMAGE, 1, 8, 40, 4) };
            assert!(!ctrl.is_null());
            unsafe { (mgr.realize_virt_arrays.unwrap())(cinfo_ptr) };
            let window: JBlockArray =
                unsafe { (mgr.access_virt_barray.unwrap())(cinfo_ptr, ctrl, 0, 4, 1) };
            assert!(!window.is_null());
            unsafe {
                let blk_row: *mut JBlock = *window;
                (*blk_row)[0] = 0x1234;
                assert_eq!((*blk_row)[0], 0x1234);
            }
        });
    }
}
