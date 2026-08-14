//! P4-165: the YUV entry points must size one plane for `TJSAMP_GRAY`.
//!
//! `tj3YUVBufSize(w, align, h, TJSAMP_GRAY)` returns a *one-plane* length, and
//! every YUV entry point is documented against that number. This port sized
//! three: `subsamp_from_tj` mapped `TJSAMP_GRAY` onto `Subsampling::S444` —
//! correct for plane 0's dimensions, wrong for the plane count — and every
//! consumer then trusted the subsampling alone to imply how many planes there
//! were.
//!
//! The consequences are all memory the caller never allocated. `tj3EncodeYUV8`
//! copied a three-plane packed length into a buffer legally sized at one
//! (an out-of-bounds *write*); `tj3CompressFromYUV8` and `tj3DecodeYUV8` built
//! a `slice::from_raw_parts` roughly three times the caller's buffer (an
//! out-of-bounds *read*); and the `…Planes8` variants walked slots 1 and 2 of
//! a plane array upstream lets the caller size at one element.
//!
//! Nothing here relies on an actual out-of-bounds access to observe that.
//! Destinations are over-allocated and prefilled with a guard byte past the
//! one-plane contract length; sources are over-allocated and filled with a
//! sentinel byte past it, and the chroma slots of every plane array point at
//! owned sentinel buffers rather than at NULL. A correct implementation leaves
//! the guards clean and never folds a sentinel into its output. Every byte
//! this test reads or writes belongs to the test process, so the assertions
//! hold whether or not the port is fixed.
//!
//! The expected trace comes from real TurboJPEG rather than from a reading of
//! `turbojpeg.c`: upstream spreads the rule across the packed wrappers
//! (`planes[1] = planes[2] = NULL` for GRAY) and the `…Planes8` workers (which
//! loop over `cinfo->num_components`, already 1), so no single site states it.
//! See `examples/tj3_yuv_gray_oracle.c`.

use std::ffi::{c_int, c_void};
use std::path::{Path, PathBuf};

mod helpers;

use libjpeg_turbo_rs_capi::{
    tj3Compress8, tj3CompressFromYUV8, tj3CompressFromYUVPlanes8, tj3DecodeYUV8,
    tj3DecodeYUVPlanes8, tj3Decompress8, tj3DecompressHeader, tj3DecompressToYUV8,
    tj3DecompressToYUVPlanes8, tj3Destroy, tj3EncodeYUV8, tj3EncodeYUVPlanes8, tj3Free, tj3Get,
    tj3Init, tj3Set, tj3YUVBufSize, tj3YUVPlaneHeight, tj3YUVPlaneWidth,
};

/// `turbojpeg.h:91-105`: `TJINIT_COMPRESS` / `TJINIT_DECOMPRESS`, a plain enum.
const TJINIT_COMPRESS: c_int = 0;
const TJINIT_DECOMPRESS: c_int = 1;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
const TJSAMP_444: c_int = 0;
const TJSAMP_420: c_int = 2;
const TJSAMP_GRAY: c_int = 3;
const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;

/// Fills the destination bytes past the one-plane contract length.
const GUARD_BYTE: u8 = 0x5A;
/// Fills the source bytes past the one-plane contract length.
const SENTINEL_BYTE: u8 = 0xC3;
/// Headroom past the largest three-plane span any case can produce, so that a
/// three-plane write lands in memory this test owns.
const SLACK: usize = 64;
const QUALITY: c_int = 90;

#[derive(Clone, Copy)]
struct Geometry {
    label: &'static str,
    width: c_int,
    height: c_int,
    align: c_int,
}

const GEOMETRIES: [Geometry; 5] = [
    Geometry {
        label: "g64x64a1",
        width: 64,
        height: 64,
        align: 1,
    },
    Geometry {
        label: "g64x64a4",
        width: 64,
        height: 64,
        align: 4,
    },
    Geometry {
        label: "g33x17a1",
        width: 33,
        height: 17,
        align: 1,
    },
    Geometry {
        label: "g33x17a4",
        width: 33,
        height: 17,
        align: 4,
    },
    Geometry {
        label: "g17x33a1",
        width: 17,
        height: 33,
        align: 1,
    },
];

impl Geometry {
    fn w(&self) -> usize {
        self.width as usize
    }

    fn h(&self) -> usize {
        self.height as usize
    }

    /// The contract length: what `tj3YUVBufSize` promises a GRAY caller.
    fn gray_size(&self) -> usize {
        tj3YUVBufSize(self.width, self.align, self.height, TJSAMP_GRAY)
    }

    /// The three-plane length a 4:4:4 caller would need — exactly what a worker
    /// that mapped GRAY onto 4:4:4 computes — plus headroom.
    fn capacity(&self) -> usize {
        tj3YUVBufSize(self.width, self.align, self.height, TJSAMP_444) + SLACK
    }

    /// Capacity for one chroma slot of a plane array, sized the way a
    /// three-plane worker would size it under 4:4:4 (full resolution).
    fn chroma_capacity(&self) -> usize {
        self.w() * self.h() + SLACK
    }

    fn packed_stride(&self) -> usize {
        pad_up(self.w(), self.align as usize)
    }
}

fn pad_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

/// FNV-1a, 64-bit — byte for byte the oracle's `fnv1a`.
fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

const FNV_OFFSET: u64 = 14695981039346656037;

/// Digest of `height` rows of `width` bytes read at `stride`.
///
/// The inter-row alignment padding is excluded deliberately: upstream never
/// writes those bytes (its row copies are `pw` wide even though its row
/// pointers advance by the stride) while this port zero-fills them. That
/// difference is about bytes inside the caller's own buffer that upstream
/// documents nothing for, and folding it in here would report it as a
/// plane-count failure.
fn row_digest(plane: &[u8], width: usize, height: usize, stride: usize) -> u64 {
    let mut hash: u64 = FNV_OFFSET;
    for row in 0..height {
        hash = fnv1a(&plane[row * stride..row * stride + width], hash);
    }
    hash
}

/// `"clean"` when every byte in `[from, to)` still holds [`GUARD_BYTE`].
fn guard_state(buffer: &[u8], from: usize, to: usize) -> &'static str {
    if buffer[from..to].iter().all(|byte| *byte == GUARD_BYTE) {
        "clean"
    } else {
        "DIRTY"
    }
}

/// The luma test pattern — non-constant in both axes, so a plane written at
/// the wrong stride, or transposed, changes the digest.
fn y_sample(x: usize, y: usize) -> u8 {
    ((x * 7 + y * 13) & 0xFF) as u8
}

/// The packed-pixel test pattern for the encode direction.
fn rgb_sample(x: usize, y: usize) -> [u8; 3] {
    [
        ((x * 5 + y * 3) & 0xFF) as u8,
        ((x * 11 + y * 7) & 0xFF) as u8,
        ((x * 3 + y * 17) & 0xFF) as u8,
    ]
}

fn make_rgb(geo: &Geometry) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0; geo.w() * geo.h() * 3];
    for y in 0..geo.h() {
        for x in 0..geo.w() {
            let base: usize = (y * geo.w() + x) * 3;
            pixels[base..base + 3].copy_from_slice(&rgb_sample(x, y));
        }
    }
    pixels
}

/// A `total`-byte buffer whose first `PAD(width, align) * height` bytes are the
/// packed GRAY plane the contract describes, and whose remaining bytes are
/// [`SENTINEL_BYTE`]. The alignment padding inside each row is sentinel too:
/// upstream skips it, so a reader that folds it into the image is as wrong as
/// one that runs past the end.
fn make_packed_gray_source(geo: &Geometry, total: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = vec![SENTINEL_BYTE; total];
    let stride: usize = geo.packed_stride();
    for y in 0..geo.h() {
        for x in 0..geo.w() {
            buffer[y * stride + x] = y_sample(x, y);
        }
    }
    buffer
}

/// The chroma test patterns, for the three-plane control cases below.
fn cb_sample(x: usize, y: usize) -> u8 {
    ((x * 3 + y * 29) & 0xFF) as u8
}

fn cr_sample(x: usize, y: usize) -> u8 {
    ((x * 23 + y * 5) & 0xFF) as u8
}

/// A `total`-byte buffer holding a full three-plane packed YUV image at
/// `subsamp`, with [`SENTINEL_BYTE`] past the planes and in each row's
/// alignment padding.
///
/// These feed the control cases: the packed entry points now route through the
/// *planes* functions for every subsampling, not only for GRAY, so the
/// three-plane path needs pinning against stock too. Without it the reroute
/// rests on a reading of the two functions rather than on a measurement.
fn make_packed_ycbcr_source(geo: &Geometry, subsamp: c_int, total: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = vec![SENTINEL_BYTE; total];
    let mut offset: usize = 0;
    for component in 0..3i32 {
        let pw: usize = tj3YUVPlaneWidth(component, geo.width, subsamp) as usize;
        let ph: usize = tj3YUVPlaneHeight(component, geo.height, subsamp) as usize;
        let stride: usize = pad_up(pw, geo.align as usize);
        for y in 0..ph {
            for x in 0..pw {
                buffer[offset + y * stride + x] = match component {
                    0 => y_sample(x, y),
                    1 => cb_sample(x, y),
                    _ => cr_sample(x, y),
                };
            }
        }
        offset += stride * ph;
    }
    buffer
}

/// The same plane, tightly packed, for the `…Planes8` entry points called with
/// a NULL stride array.
fn make_tight_gray_plane(geo: &Geometry) -> Vec<u8> {
    let mut plane: Vec<u8> = vec![0; geo.w() * geo.h()];
    for y in 0..geo.h() {
        for x in 0..geo.w() {
            plane[y * geo.w() + x] = y_sample(x, y);
        }
    }
    plane
}

/// `"yes"` when every pixel of a packed RGB buffer has R == G == B.
///
/// The structural signature of a grayscale decode: YCbCr → RGB with
/// Cb = Cr = 128 collapses to R = G = B = Y, and nothing else does. A decoder
/// that read sentinel bytes as chroma produces colour here whatever its digest.
fn rgb_is_gray(pixels: &[u8], width: usize, height: usize) -> &'static str {
    let gray: bool = (0..width * height)
        .all(|i| pixels[i * 3] == pixels[i * 3 + 1] && pixels[i * 3] == pixels[i * 3 + 2]);
    if gray {
        "yes"
    } else {
        "no"
    }
}

/// How faithfully a JPEG produced from the GRAY plane decodes back to it.
///
/// Exact JPEG bytes are not comparable across two independent encoders, so the
/// compress-from-YUV cases assert the round trip instead: `"ok"` means every
/// sample came back within 16 of the source. Measured at 9-12 across the 10
/// compress cases here (stock, quality 90, a gradient that is deliberately
/// hostile to an 8×8 DCT); 16 is that worst case plus a small margin, so a
/// quantisation difference between two encoders cannot read as a plane-count
/// failure, which is the contract actually being pinned. A number in place of
/// `"ok"` is a real divergence — chroma folded into the luma plane moves this
/// into the hundreds.
fn roundtrip_state(decoded: &[u8], width: usize, height: usize) -> String {
    let mut worst: i32 = 0;
    for y in 0..height {
        for x in 0..width {
            let diff: i32 = decoded[y * width + x] as i32 - y_sample(x, y) as i32;
            worst = worst.max(diff.abs());
        }
    }
    if worst <= 16 {
        "ok".to_string()
    } else {
        format!("bad({worst})")
    }
}

/// A handle, or a panic naming the call that failed — a NULL handle here is a
/// failure of this crate, never a reason to skip.
fn instance(init_type: c_int) -> *mut c_void {
    let handle: *mut c_void = tj3Init(init_type);
    assert!(!handle.is_null(), "tj3Init({init_type})");
    handle
}

fn set_param(handle: *mut c_void, param: c_int, value: c_int) {
    // SAFETY: `handle` is a live instance used exclusively by this thread.
    let rc: c_int = unsafe { tj3Set(handle, param, value) };
    assert_eq!(rc, 0, "tj3Set({param}, {value})");
}

// --- Pixels → YUV --------------------------------------------------------

fn case_encode_yuv8(geo: &Geometry) -> String {
    let handle: *mut c_void = instance(TJINIT_COMPRESS);
    let pixels: Vec<u8> = make_rgb(geo);
    let mut dst: Vec<u8> = vec![GUARD_BYTE; geo.capacity()];
    set_param(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    // SAFETY: `pixels` is `width * height * 3` bytes as the pitch-0 contract
    // requires, and `dst` is over-allocated past the one-plane length so a
    // three-plane write stays inside this test's own allocation.
    let rc: c_int = unsafe {
        tj3EncodeYUV8(
            handle,
            pixels.as_ptr(),
            geo.width,
            0,
            geo.height,
            TJPF_RGB,
            dst.as_mut_ptr(),
            geo.align,
        )
    };
    let line: String = format!(
        "{} encodeyuv8 rc={rc} ydigest={:016x} guard={}\n",
        geo.label,
        row_digest(&dst, geo.w(), geo.h(), geo.packed_stride()),
        guard_state(&dst, geo.gray_size(), geo.capacity()),
    );
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

fn case_encode_yuv_planes8(geo: &Geometry) -> String {
    let handle: *mut c_void = instance(TJINIT_COMPRESS);
    let pixels: Vec<u8> = make_rgb(geo);
    let mut luma: Vec<u8> = vec![GUARD_BYTE; geo.w() * geo.h()];
    // Slots 1 and 2 hold real, owned buffers rather than NULL so that a worker
    // which walks all three planes writes here instead of running off the end
    // of a one-element array. Upstream accepts either for GRAY: its NULL check
    // is `subsamp != TJSAMP_GRAY && (!dstPlanes[1] || !dstPlanes[2])`
    // (`turbojpeg.c:1589-1590`).
    let mut chroma0: Vec<u8> = vec![GUARD_BYTE; geo.chroma_capacity()];
    let mut chroma1: Vec<u8> = vec![GUARD_BYTE; geo.chroma_capacity()];
    let mut planes: [*mut u8; 3] = [
        luma.as_mut_ptr(),
        chroma0.as_mut_ptr(),
        chroma1.as_mut_ptr(),
    ];
    set_param(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    // SAFETY: three live plane pointers into buffers this test owns, each
    // large enough for the full-resolution plane a 4:4:4 worker would write;
    // a NULL stride array means tightly packed rows.
    let rc: c_int = unsafe {
        tj3EncodeYUVPlanes8(
            handle,
            pixels.as_ptr(),
            geo.width,
            0,
            geo.height,
            TJPF_RGB,
            planes.as_mut_ptr(),
            std::ptr::null(),
        )
    };
    let line: String = format!(
        "{} encodeplanes8 rc={rc} ydigest={:016x} guard1={} guard2={}\n",
        geo.label,
        row_digest(&luma, geo.w(), geo.h(), geo.w()),
        guard_state(&chroma0, 0, geo.chroma_capacity()),
        guard_state(&chroma1, 0, geo.chroma_capacity()),
    );
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

// --- YUV → pixels --------------------------------------------------------

fn case_decode_yuv8(geo: &Geometry) -> String {
    let handle: *mut c_void = instance(TJINIT_DECOMPRESS);
    let src: Vec<u8> = make_packed_gray_source(geo, geo.capacity());
    let mut pixels: Vec<u8> = vec![GUARD_BYTE; geo.w() * geo.h() * 3];
    set_param(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    // SAFETY: `src` is over-allocated past the one-plane length, so a reader
    // that takes three planes still stays inside this test's allocation; the
    // destination is exactly the pitch-0 packed-RGB size.
    let rc: c_int = unsafe {
        tj3DecodeYUV8(
            handle,
            src.as_ptr(),
            geo.align,
            pixels.as_mut_ptr(),
            geo.width,
            0,
            geo.height,
            TJPF_RGB,
        )
    };
    let line: String = format!(
        "{} decodeyuv8 rc={rc} rgbdigest={:016x} rgbgray={}\n",
        geo.label,
        row_digest(&pixels, geo.w() * 3, geo.h(), geo.w() * 3),
        rgb_is_gray(&pixels, geo.w(), geo.h()),
    );
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

fn case_decode_yuv_planes8(geo: &Geometry) -> String {
    let handle: *mut c_void = instance(TJINIT_DECOMPRESS);
    let luma: Vec<u8> = make_tight_gray_plane(geo);
    let chroma0: Vec<u8> = vec![SENTINEL_BYTE; geo.chroma_capacity()];
    let chroma1: Vec<u8> = vec![SENTINEL_BYTE; geo.chroma_capacity()];
    let planes: [*const u8; 3] = [luma.as_ptr(), chroma0.as_ptr(), chroma1.as_ptr()];
    let mut pixels: Vec<u8> = vec![GUARD_BYTE; geo.w() * geo.h() * 3];
    set_param(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    // SAFETY: three live plane pointers into buffers this test owns; the
    // chroma slots are full-resolution, which is what a 4:4:4 worker reads.
    let rc: c_int = unsafe {
        tj3DecodeYUVPlanes8(
            handle,
            planes.as_ptr(),
            std::ptr::null(),
            pixels.as_mut_ptr(),
            geo.width,
            0,
            geo.height,
            TJPF_RGB,
        )
    };
    let line: String = format!(
        "{} decodeplanes8 rc={rc} rgbdigest={:016x} rgbgray={}\n",
        geo.label,
        row_digest(&pixels, geo.w() * 3, geo.h(), geo.w() * 3),
        rgb_is_gray(&pixels, geo.w(), geo.h()),
    );
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

// --- YUV → JPEG ----------------------------------------------------------

/// Report a compressed result by what a caller can check about it: the header
/// a decompressor reads back, and whether the image survived the round trip.
/// Not by its bytes — two encoders agreeing on JPEG output is a stronger claim
/// than this item is about, and one this port does not owe on the YUV path.
fn report_compressed(geo: &Geometry, case_name: &str, rc: c_int, jpeg: &[u8]) -> String {
    if rc != 0 || jpeg.is_empty() {
        return format!(
            "{} {case_name} rc={rc} jw=0 jh=0 jsubsamp=-2 roundtrip=none\n",
            geo.label
        );
    }
    let handle: *mut c_void = instance(TJINIT_DECOMPRESS);
    // SAFETY: `jpeg` is a live slice for the duration of the call.
    let header_rc: c_int = unsafe { tj3DecompressHeader(handle, jpeg.as_ptr(), jpeg.len()) };
    if header_rc != 0 {
        // SAFETY: `handle` came from `tj3Init` and is not used again.
        unsafe { tj3Destroy(handle) };
        return format!(
            "{} {case_name} rc={rc} jw=0 jh=0 jsubsamp=-3 roundtrip=none\n",
            geo.label
        );
    }
    let mut decoded: Vec<u8> = vec![GUARD_BYTE; geo.w() * geo.h()];
    // SAFETY: `decoded` is `width * height`, the pitch-0 size for TJPF_GRAY at
    // the header dimensions asserted below.
    let decode_rc: c_int = unsafe {
        tj3Decompress8(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            decoded.as_mut_ptr(),
            0,
            TJPF_GRAY,
        )
    };
    // SAFETY: `handle` is live until the `tj3Destroy` below.
    let (jw, jh, subsamp): (c_int, c_int, c_int) = unsafe {
        (
            tj3Get(handle, TJPARAM_JPEGWIDTH),
            tj3Get(handle, TJPARAM_JPEGHEIGHT),
            tj3Get(handle, TJPARAM_SUBSAMP),
        )
    };
    let roundtrip: String = if decode_rc != 0 {
        "undecodable".to_string()
    } else {
        roundtrip_state(&decoded, geo.w(), geo.h())
    };
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    format!(
        "{} {case_name} rc={rc} jw={jw} jh={jh} jsubsamp={subsamp} roundtrip={roundtrip}\n",
        geo.label
    )
}

/// The library-allocated compressed output of one call, copied out and freed.
///
/// # Safety
///
/// `jpeg` must be the pointer a TurboJPEG compress entry point wrote, and
/// `size` its reported length.
unsafe fn take_output(jpeg: *mut u8, size: usize) -> Vec<u8> {
    if jpeg.is_null() || size == 0 {
        return Vec::new();
    }
    // SAFETY: the caller guarantees `jpeg` is valid for `size` bytes, and the
    // compress entry point transferred ownership of the allocation.
    let bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(jpeg, size) }.to_vec();
    // SAFETY: same allocation, released exactly once.
    unsafe { tj3Free(jpeg as *mut c_void) };
    bytes
}

fn case_compress_from_yuv8(geo: &Geometry) -> String {
    let handle: *mut c_void = instance(TJINIT_COMPRESS);
    let src: Vec<u8> = make_packed_gray_source(geo, geo.capacity());
    let mut jpeg: *mut u8 = std::ptr::null_mut();
    let mut jpeg_size: usize = 0;
    set_param(handle, TJPARAM_QUALITY, QUALITY);
    set_param(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    // SAFETY: `src` is over-allocated past the one-plane length; the output
    // slot is a null pointer plus a zero size, which asks the library to
    // allocate.
    let rc: c_int = unsafe {
        tj3CompressFromYUV8(
            handle,
            src.as_ptr(),
            geo.width,
            geo.align,
            geo.height,
            &mut jpeg,
            &mut jpeg_size,
        )
    };
    // SAFETY: `jpeg`/`jpeg_size` are exactly what the call above reported.
    let output: Vec<u8> = unsafe { take_output(jpeg, jpeg_size) };
    let line: String = report_compressed(geo, "fromyuv8", rc, &output);
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

fn case_compress_from_yuv_planes8(geo: &Geometry) -> String {
    let handle: *mut c_void = instance(TJINIT_COMPRESS);
    let luma: Vec<u8> = make_tight_gray_plane(geo);
    let chroma0: Vec<u8> = vec![SENTINEL_BYTE; geo.chroma_capacity()];
    let chroma1: Vec<u8> = vec![SENTINEL_BYTE; geo.chroma_capacity()];
    let planes: [*const u8; 3] = [luma.as_ptr(), chroma0.as_ptr(), chroma1.as_ptr()];
    let mut jpeg: *mut u8 = std::ptr::null_mut();
    let mut jpeg_size: usize = 0;
    set_param(handle, TJPARAM_QUALITY, QUALITY);
    set_param(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    // SAFETY: three live plane pointers into buffers this test owns, sized the
    // way a 4:4:4 worker would read them.
    let rc: c_int = unsafe {
        tj3CompressFromYUVPlanes8(
            handle,
            planes.as_ptr(),
            geo.width,
            std::ptr::null(),
            geo.height,
            &mut jpeg,
            &mut jpeg_size,
        )
    };
    // SAFETY: `jpeg`/`jpeg_size` are exactly what the call above reported.
    let output: Vec<u8> = unsafe { take_output(jpeg, jpeg_size) };
    let line: String = report_compressed(geo, "fromyuvplanes8", rc, &output);
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

// --- JPEG → YUV ----------------------------------------------------------

fn case_decompress_to_yuv8(geo: &Geometry, jpeg: &[u8]) -> String {
    let handle: *mut c_void = instance(TJINIT_DECOMPRESS);
    let mut dst: Vec<u8> = vec![GUARD_BYTE; geo.capacity()];
    // SAFETY: `dst` is over-allocated past the one-plane length so a
    // three-plane write stays inside this test's own allocation.
    let rc: c_int = unsafe {
        tj3DecompressToYUV8(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            dst.as_mut_ptr(),
            geo.align,
        )
    };
    let line: String = format!(
        "{} toyuv8 rc={rc} ydigest={:016x} guard={}\n",
        geo.label,
        row_digest(&dst, geo.w(), geo.h(), geo.packed_stride()),
        guard_state(&dst, geo.gray_size(), geo.capacity()),
    );
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

fn case_decompress_to_yuv_planes8(geo: &Geometry, jpeg: &[u8]) -> String {
    let handle: *mut c_void = instance(TJINIT_DECOMPRESS);
    let mut luma: Vec<u8> = vec![GUARD_BYTE; geo.w() * geo.h()];
    let mut chroma0: Vec<u8> = vec![GUARD_BYTE; geo.chroma_capacity()];
    let mut chroma1: Vec<u8> = vec![GUARD_BYTE; geo.chroma_capacity()];
    let mut planes: [*mut u8; 3] = [
        luma.as_mut_ptr(),
        chroma0.as_mut_ptr(),
        chroma1.as_mut_ptr(),
    ];
    // SAFETY: three live plane pointers into buffers this test owns, each
    // large enough for the full-resolution plane a 4:4:4 worker would write.
    let rc: c_int = unsafe {
        tj3DecompressToYUVPlanes8(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            planes.as_mut_ptr(),
            std::ptr::null(),
        )
    };
    let line: String = format!(
        "{} toyuvplanes8 rc={rc} ydigest={:016x} guard1={} guard2={}\n",
        geo.label,
        row_digest(&luma, geo.w(), geo.h(), geo.w()),
        guard_state(&chroma0, 0, geo.chroma_capacity()),
        guard_state(&chroma1, 0, geo.chroma_capacity()),
    );
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

// --- Three-plane controls ------------------------------------------------
//
// Not about TJSAMP_GRAY: these pin the *other* subsamplings through the same
// entry points, because the packed paths stopped inferring their plane count
// from the buffer's length and now split the buffer themselves before calling
// the planar workers. That reroute touches every subsampling, and the GRAY
// cases above cannot see it. `TJPF_GRAY` is included because it is the one
// output format whose internal branch changed — a three-plane source decoded
// to grayscale used to take a single-plane shortcut here.

fn case_decode_yuv8_three_plane(
    geo: &Geometry,
    subsamp: c_int,
    pixel_format: c_int,
    case_name: &str,
    bpp: usize,
) -> String {
    let handle: *mut c_void = instance(TJINIT_DECOMPRESS);
    let source_size: usize = tj3YUVBufSize(geo.width, geo.align, geo.height, subsamp);
    let src: Vec<u8> = make_packed_ycbcr_source(geo, subsamp, source_size + SLACK);
    let mut pixels: Vec<u8> = vec![GUARD_BYTE; geo.w() * geo.h() * bpp];
    set_param(handle, TJPARAM_SUBSAMP, subsamp);
    // SAFETY: `src` holds a full three-plane image plus sentinel headroom, and
    // `pixels` is the pitch-0 packed size for `pixel_format`.
    let rc: c_int = unsafe {
        tj3DecodeYUV8(
            handle,
            src.as_ptr(),
            geo.align,
            pixels.as_mut_ptr(),
            geo.width,
            0,
            geo.height,
            pixel_format,
        )
    };
    let line: String = format!(
        "{} {case_name} rc={rc} digest={:016x}\n",
        geo.label,
        row_digest(&pixels, geo.w() * bpp, geo.h(), geo.w() * bpp),
    );
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    line
}

/// `(rc, jpeg)` from `tj3CompressFromYUV8` over a three-plane packed image.
fn compress_from_yuv8_three_plane(geo: &Geometry, subsamp: c_int) -> (c_int, Vec<u8>) {
    let handle: *mut c_void = instance(TJINIT_COMPRESS);
    let source_size: usize = tj3YUVBufSize(geo.width, geo.align, geo.height, subsamp);
    let src: Vec<u8> = make_packed_ycbcr_source(geo, subsamp, source_size + SLACK);
    let mut jpeg: *mut u8 = std::ptr::null_mut();
    let mut jpeg_size: usize = 0;
    set_param(handle, TJPARAM_QUALITY, QUALITY);
    set_param(handle, TJPARAM_SUBSAMP, subsamp);
    // SAFETY: `src` holds a full three-plane image plus sentinel headroom; the
    // output slot is a null pointer plus a zero size, asking the library to
    // allocate.
    let rc: c_int = unsafe {
        tj3CompressFromYUV8(
            handle,
            src.as_ptr(),
            geo.width,
            geo.align,
            geo.height,
            &mut jpeg,
            &mut jpeg_size,
        )
    };
    // SAFETY: `jpeg`/`jpeg_size` are exactly what the call above reported.
    let output: Vec<u8> = unsafe { take_output(jpeg, jpeg_size) };
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    (rc, output)
}

/// `(rc, jpeg)` from `tj3CompressFromYUVPlanes8` over the same image, as three
/// tightly packed planes.
fn compress_from_yuv_planes8_three_plane(geo: &Geometry, subsamp: c_int) -> (c_int, Vec<u8>) {
    let handle: *mut c_void = instance(TJINIT_COMPRESS);
    let dense: Vec<Vec<u8>> = (0..3i32)
        .map(|component| {
            let pw: usize = tj3YUVPlaneWidth(component, geo.width, subsamp) as usize;
            let ph: usize = tj3YUVPlaneHeight(component, geo.height, subsamp) as usize;
            let mut plane: Vec<u8> = vec![0; pw * ph];
            for y in 0..ph {
                for x in 0..pw {
                    plane[y * pw + x] = match component {
                        0 => y_sample(x, y),
                        1 => cb_sample(x, y),
                        _ => cr_sample(x, y),
                    };
                }
            }
            plane
        })
        .collect();
    let planes: [*const u8; 3] = [dense[0].as_ptr(), dense[1].as_ptr(), dense[2].as_ptr()];
    let mut jpeg: *mut u8 = std::ptr::null_mut();
    let mut jpeg_size: usize = 0;
    set_param(handle, TJPARAM_QUALITY, QUALITY);
    set_param(handle, TJPARAM_SUBSAMP, subsamp);
    // SAFETY: three live plane pointers into buffers this test owns, each the
    // full `tj3YUVPlaneWidth × tj3YUVPlaneHeight`; a NULL stride array means
    // tightly packed rows.
    let rc: c_int = unsafe {
        tj3CompressFromYUVPlanes8(
            handle,
            planes.as_ptr(),
            geo.width,
            std::ptr::null(),
            geo.height,
            &mut jpeg,
            &mut jpeg_size,
        )
    };
    // SAFETY: `jpeg`/`jpeg_size` are exactly what the call above reported.
    let output: Vec<u8> = unsafe { take_output(jpeg, jpeg_size) };
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    (rc, output)
}

/// A grayscale JPEG of this geometry, encoded here rather than read from the
/// oracle's workdir. Used only by the standalone assertions, which must keep
/// their teeth on a machine with no TurboJPEG development install.
fn our_gray_jpeg(geo: &Geometry) -> Vec<u8> {
    let handle: *mut c_void = instance(TJINIT_COMPRESS);
    let gray: Vec<u8> = make_tight_gray_plane(geo);
    let mut jpeg: *mut u8 = std::ptr::null_mut();
    let mut jpeg_size: usize = 0;
    set_param(handle, TJPARAM_QUALITY, QUALITY);
    set_param(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    // SAFETY: `gray` is `width * height`, the pitch-0 size for TJPF_GRAY.
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            gray.as_ptr(),
            geo.width,
            0,
            geo.height,
            TJPF_GRAY,
            &mut jpeg,
            &mut jpeg_size,
        )
    };
    assert_eq!(rc, 0, "tj3Compress8 of a grayscale {} image", geo.label);
    // SAFETY: `jpeg`/`jpeg_size` are exactly what the call above reported.
    let output: Vec<u8> = unsafe { take_output(jpeg, jpeg_size) };
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
    assert!(!output.is_empty(), "empty grayscale JPEG for {}", geo.label);
    output
}

/// Every case the oracle traces, in its order, as one string.
///
/// `fixtures` supplies the grayscale JPEG for each geometry. The oracle writes
/// those files, so both implementations decompress identical bytes rather than
/// whatever each one's own encoder produced.
fn our_trace(fixtures: &dyn Fn(&Geometry) -> Vec<u8>) -> String {
    let mut trace: String = String::new();
    for geo in &GEOMETRIES {
        trace.push_str(&format!(
            "{} bufsize={} capacity={}\n",
            geo.label,
            geo.gray_size(),
            geo.capacity()
        ));
        trace.push_str(&case_encode_yuv8(geo));
        trace.push_str(&case_encode_yuv_planes8(geo));
        trace.push_str(&case_decode_yuv8(geo));
        trace.push_str(&case_decode_yuv_planes8(geo));
        trace.push_str(&case_compress_from_yuv8(geo));
        trace.push_str(&case_compress_from_yuv_planes8(geo));
        let jpeg: Vec<u8> = fixtures(geo);
        trace.push_str(&case_decompress_to_yuv8(geo, &jpeg));
        trace.push_str(&case_decompress_to_yuv_planes8(geo, &jpeg));

        // Three-plane controls for the packed reroute.
        trace.push_str(&case_decode_yuv8_three_plane(
            geo,
            TJSAMP_420,
            TJPF_RGB,
            "decodeyuv8_420rgb",
            3,
        ));
        trace.push_str(&case_decode_yuv8_three_plane(
            geo,
            TJSAMP_420,
            TJPF_GRAY,
            "decodeyuv8_420gray",
            1,
        ));
    }
    trace
}

/// The whole trace, compared verbatim against real TurboJPEG.
#[test]
fn gray_yuv_entry_points_match_stock_turbojpeg() {
    let Some(oracle) = helpers::build_oracle("tj3_yuv_gray_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; the C oracle for \
             P4-165's GRAY plane count cannot be built. Set LIBJPEG_TURBO_PREFIX \
             to make this a hard failure."
        );
        return;
    };
    let workdir: PathBuf =
        std::env::temp_dir().join(format!("libjpeg_turbo_rs_p4165_{}", std::process::id()));
    std::fs::create_dir_all(&workdir).expect("create oracle workdir");

    let c_trace: String = helpers::run_oracle(&oracle, &[workdir.to_str().expect("utf-8 workdir")]);

    let read_fixture = |geo: &Geometry| -> Vec<u8> {
        let path: PathBuf = workdir.join(format!("{}.jpg", geo.label));
        std::fs::read(&path)
            .unwrap_or_else(|e| panic!("oracle fixture {} unreadable: {e}", path.display()))
    };
    let rust_trace: String = our_trace(&read_fixture);

    let _ = std::fs::remove_dir_all(&workdir);
    assert_eq!(
        rust_trace, c_trace,
        "the TJSAMP_GRAY YUV paths diverge from upstream TurboJPEG"
    );
}

// --- The divergence itself, stated without reference to the oracle -------
//
// These keep the gate's teeth on a machine with no TurboJPEG development
// install. Each asserts one half of the plane-count contract directly, so a
// regression fails here even when the comparison above skips.

/// One line of `our_trace` for one geometry, located by its case name.
fn line_for(trace: &str, geo: &Geometry, case_name: &str) -> String {
    let prefix: String = format!("{} {case_name} ", geo.label);
    trace
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in:\n{trace}"))
        .to_string()
}

#[test]
fn packed_gray_encode_writes_only_the_luma_plane() {
    for geo in &GEOMETRIES {
        let line: String = case_encode_yuv8(geo);
        assert!(
            line.contains(" rc=0 ") && line.ends_with(" guard=clean\n"),
            "tj3EncodeYUV8 must write at most tj3YUVBufSize(.., TJSAMP_GRAY) bytes; \
             a three-plane packed length overruns a buffer the caller sized from \
             the documented contract: {line}"
        );
    }
}

#[test]
fn planar_gray_encode_touches_only_plane_zero() {
    for geo in &GEOMETRIES {
        let line: String = case_encode_yuv_planes8(geo);
        assert!(
            line.contains(" rc=0 ") && line.ends_with(" guard1=clean guard2=clean\n"),
            "tj3EncodeYUVPlanes8 must not write dstPlanes[1]/[2] for TJSAMP_GRAY; \
             upstream passes NULL there, so a caller may legally supply a \
             one-element array: {line}"
        );
    }
}

#[test]
fn packed_gray_decode_ignores_bytes_past_the_luma_plane() {
    for geo in &GEOMETRIES {
        let line: String = case_decode_yuv8(geo);
        assert!(
            line.contains(" rc=0 ") && line.ends_with(" rgbgray=yes\n"),
            "tj3DecodeYUV8 must read only tj3YUVBufSize(.., TJSAMP_GRAY) bytes; \
             folding the sentinel tail in as chroma both overruns the caller's \
             buffer and colours a grayscale image: {line}"
        );
    }
}

#[test]
fn planar_gray_decode_ignores_the_chroma_slots() {
    for geo in &GEOMETRIES {
        let line: String = case_decode_yuv_planes8(geo);
        assert!(
            line.contains(" rc=0 ") && line.ends_with(" rgbgray=yes\n"),
            "tj3DecodeYUVPlanes8 must not read srcPlanes[1]/[2] for TJSAMP_GRAY: {line}"
        );
    }
}

#[test]
fn gray_compress_from_yuv_emits_a_one_component_jpeg() {
    for geo in &GEOMETRIES {
        for (case_name, line) in [
            ("fromyuv8", case_compress_from_yuv8(geo)),
            ("fromyuvplanes8", case_compress_from_yuv_planes8(geo)),
        ] {
            assert_eq!(
                line,
                format!(
                    "{} {case_name} rc=0 jw={} jh={} jsubsamp={TJSAMP_GRAY} roundtrip=ok\n",
                    geo.label, geo.width, geo.height
                ),
                "a TJSAMP_GRAY compress must produce a grayscale JPEG that round-trips \
                 to the source luma; a three-component stream means chroma was \
                 invented from bytes past the caller's buffer"
            );
        }
    }
}

/// The decompress direction, which was already correct and stays that way.
///
/// Unlike the five above, this one does not move when the plane-count
/// mechanism is broken: `tj3DecompressToYUV*8` takes its count from the JPEG's
/// own SOF rather than from `TJPARAM_SUBSAMP`, exactly as upstream does, so it
/// never consulted the mapping this item repairs. It is a regression pin, not
/// a reproduction — kept because a future refactor that routed all eight entry
/// points through one plane-count helper could break it while the other five
/// stay green.
#[test]
fn gray_decompress_to_yuv_emits_only_the_luma_plane() {
    for geo in &GEOMETRIES {
        let jpeg: Vec<u8> = our_gray_jpeg(geo);
        let packed: String = case_decompress_to_yuv8(geo, &jpeg);
        assert!(
            packed.contains(" rc=0 ") && packed.ends_with(" guard=clean\n"),
            "tj3DecompressToYUV8 of a grayscale JPEG must fill one plane: {packed}"
        );
        let planar: String = case_decompress_to_yuv_planes8(geo, &jpeg);
        assert!(
            planar.contains(" rc=0 ") && planar.ends_with(" guard1=clean guard2=clean\n"),
            "tj3DecompressToYUVPlanes8 of a grayscale JPEG must fill one plane: {planar}"
        );
    }
}

/// The packed and planar spellings of the same operation must agree.
///
/// Upstream guarantees this by construction — each packed wrapper computes
/// plane pointers and calls its `…Planes8` sibling — while this port
/// implements the two separately, so nothing but a test holds them together.
#[test]
fn packed_and_planar_gray_paths_agree() {
    let jpegs: Vec<Vec<u8>> = GEOMETRIES.iter().map(our_gray_jpeg).collect();
    let trace: String = our_trace(&|geo: &Geometry| {
        let index: usize = GEOMETRIES
            .iter()
            .position(|candidate| candidate.label == geo.label)
            .expect("geometry is one of GEOMETRIES");
        jpegs[index].clone()
    });
    for geo in &GEOMETRIES {
        let digest_of = |case_name: &str| -> String {
            let line: String = line_for(&trace, geo, case_name);
            // Without this, two failed calls agree trivially: both
            // destinations are prefilled with GUARD_BYTE, so "wrote nothing"
            // and "wrote nothing" have the same digest.
            assert!(
                line.contains(" rc=0 "),
                "{} {case_name} failed, so its digest proves nothing: {line}",
                geo.label
            );
            line.split_whitespace()
                .find_map(|field| field.strip_prefix("ydigest=").map(str::to_string))
                .expect("a ydigest field")
        };
        assert_eq!(
            digest_of("encodeyuv8"),
            digest_of("encodeplanes8"),
            "{}: tj3EncodeYUV8 and tj3EncodeYUVPlanes8 produced different luma",
            geo.label
        );
        assert_eq!(
            digest_of("toyuv8"),
            digest_of("toyuvplanes8"),
            "{}: tj3DecompressToYUV8 and tj3DecompressToYUVPlanes8 produced different luma",
            geo.label
        );
    }
}

/// The packed and planar compress spellings must agree on a three-plane image.
///
/// This is the three-plane half of the reroute the GRAY cases cannot see: the
/// packed entry point used to hand its whole buffer to a function that inferred
/// the plane count from the buffer's *length*, and now splits the buffer itself
/// and calls the planar worker, the same one `tj3CompressFromYUVPlanes8` uses.
/// The two must therefore produce the same bytes. Stated as an internal
/// equivalence rather than as an oracle line because the oracle comparison
/// cannot run here: `tj3CompressFromYUV8` diverges from stock at 4:2:0 for two
/// reasons that predate this change and are tracked as P4-167 — it refuses
/// non-MCU-aligned dimensions outright, and its output cannot be decompressed
/// to `TJPF_GRAY`. 64×64 is chosen because it is MCU-aligned, so the first of
/// those does not fire.
///
/// A regression pin, not a reproduction: both spellings agreed before the
/// reroute too. It exists so the reroute cannot silently desynchronise them.
#[test]
fn packed_and_planar_three_plane_compress_agree() {
    let geo: Geometry = Geometry {
        label: "g64x64a1",
        width: 64,
        height: 64,
        align: 1,
    };
    let (packed_rc, packed_jpeg) = compress_from_yuv8_three_plane(&geo, TJSAMP_420);
    let (planar_rc, planar_jpeg) = compress_from_yuv_planes8_three_plane(&geo, TJSAMP_420);
    assert_eq!(packed_rc, 0, "tj3CompressFromYUV8 at 4:2:0");
    assert_eq!(planar_rc, 0, "tj3CompressFromYUVPlanes8 at 4:2:0");
    assert!(!packed_jpeg.is_empty(), "empty packed output");
    assert_eq!(
        packed_jpeg, planar_jpeg,
        "the packed and planar compress-from-YUV entry points produced different \
         JPEGs from the same three-plane image; they share a worker and must not"
    );
}

/// The oracle's own workdir contract, so a mis-typed path fails as a missing
/// fixture rather than as a silent empty trace.
#[test]
fn oracle_source_is_present() {
    let source: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("tj3_yuv_gray_oracle.c");
    assert!(
        source.exists(),
        "missing oracle source {}",
        source.display()
    );
}
