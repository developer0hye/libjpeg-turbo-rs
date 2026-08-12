//! P4-145 (#514): every compressing entry point must honour
//! `TJPARAM_NOREALLOC`, not just `tj3Compress8`.
//!
//! Upstream lets a caller pre-allocate the output buffer and set the flag to
//! promise the library will not resize it. That promise is what makes a
//! caller-owned buffer usable at all: the pointer the caller passed is the
//! pointer it gets back.
//!
//! Five entry points ignored the parameter — `tj3Compress12`, `tj3Compress16`,
//! `tj3CompressFromYUV8`, `tj3CompressFromYUVPlanes8`, and `tj3Transform`
//! (whose reusable slots are `dst_bufs[i]`). Each allocated a fresh buffer and
//! `free()`d the previous pointee unconditionally. Two consequences, the
//! second serious:
//!
//! 1. A caller that pre-sized its buffer got a *different* pointer back, so C
//!    code holding the original was left with a dangling one.
//! 2. `NOREALLOC` is precisely the flag a caller sets when the buffer is *not*
//!    `malloc`-owned — a stack array, or a `Vec` on the Rust side. Upstream's
//!    contract makes that safe. Ours freed it, with the wrong allocator.
//!
//! **What each test asserts is the pointer identity**, which is the observable
//! difference and the one a documentation fix could not deliver. A test that
//! only checked the encode succeeded would have passed throughout.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::{
    tj3Alloc, tj3Compress12, tj3Compress16, tj3Compress8, tj3CompressFromYUV8,
    tj3CompressFromYUVPlanes8, tj3Destroy, tj3Free, tj3Init, tj3Set, tj3Transform, TjTransform,
};

const TJINIT_COMPRESS: c_int = 0;
///  — a plain enum, so 0/1/2, not bit flags.
const TJINIT_TRANSFORM: c_int = 2;
const TJPARAM_NOREALLOC: c_int = 2;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJSAMP_444: c_int = 0;
const TJPF_RGB: c_int = 0;

const WIDTH: c_int = 32;
const HEIGHT: c_int = 32;
/// Comfortably larger than any 32x32 output here, so "fits" is never in doubt.
const ROOMY: usize = 64 * 1024;
/// Far too small for a real JPEG, but a genuine `tj3Alloc` allocation — so the
/// only thing standing between the encoder and an overrun is the capacity
/// check itself.
const CRAMPED: usize = 16;

/// A compressor handle with `NOREALLOC` set, plus quality/subsampling so the
/// output size is predictable.
fn compressor(init: c_int) -> *mut c_void {
    let handle: *mut c_void = tj3Init(init);
    assert!(!handle.is_null(), "tj3Init({init})");
    // SAFETY: `handle` is a live instance from `tj3Init`.
    unsafe {
        assert_eq!(tj3Set(handle, TJPARAM_NOREALLOC, 1), 0, "set NOREALLOC");
        if init != TJINIT_TRANSFORM {
            assert_eq!(tj3Set(handle, TJPARAM_QUALITY, 80), 0, "set quality");
            assert_eq!(
                tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444),
                0,
                "set subsamp"
            );
        }
    }
    handle
}

/// A `tj3Alloc` buffer of `bytes`, and the pointer value to compare against
/// afterwards.
fn caller_buffer(bytes: usize) -> *mut u8 {
    let buf: *mut c_void = tj3Alloc(bytes);
    assert!(!buf.is_null(), "tj3Alloc({bytes})");
    buf as *mut u8
}

fn rgb_source() -> Vec<u8> {
    (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect()
}

/// Assert the pointer survived, and free it exactly once.
fn assert_kept(label: &str, slot: *mut u8, original: *mut u8, size: usize) {
    assert_eq!(
        slot, original,
        "{label}: TJPARAM_NOREALLOC is set, so the caller's buffer pointer must \
         come back unchanged — a swapped pointer leaves C code holding a \
         dangling one, and the original is freed with the wrong allocator when \
         it was never malloc'd"
    );
    assert!(size > 0, "{label}: output size must be reported");
    assert!(size <= ROOMY, "{label}: output cannot exceed the capacity");
}

#[test]
fn tj3_compress8_keeps_the_callers_buffer() {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let src: Vec<u8> = rgb_source();
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut buf: *mut u8 = original;
    let mut size: usize = ROOMY;

    // SAFETY: live handle; `src` holds `WIDTH * HEIGHT * 3` bytes; `buf`/`size`
    // are a valid out-pair whose buffer holds `ROOMY` bytes.
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    assert_eq!(rc, 0, "tj3Compress8 must succeed");
    assert_kept("tj3Compress8", buf, original, size);

    // SAFETY: `original` came from `tj3Alloc` and is freed exactly once — the
    // point of the assertion above is that the library did not free it first.
    unsafe { tj3Free(original as *mut c_void) };
    // SAFETY: `handle` came from `tj3Init` and is destroyed once.
    unsafe { tj3Destroy(handle) };
}

#[test]
fn tj3_compress12_keeps_the_callers_buffer() {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let src: Vec<i16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 4096) as i16)
        .collect();
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut buf: *mut u8 = original;
    let mut size: usize = ROOMY;

    // SAFETY: as above, with a 12-bit source of the same geometry.
    let rc: c_int = unsafe {
        tj3Compress12(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    assert_eq!(rc, 0, "tj3Compress12 must succeed");
    assert_kept("tj3Compress12", buf, original, size);

    // SAFETY: `original` came from `tj3Alloc` and is freed exactly once — the
    // point of the assertion above is that the library did not free it first.
    unsafe { tj3Free(original as *mut c_void) };
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
}

#[test]
fn tj3_compress16_keeps_the_callers_buffer() {
    /// `turbojpeg.h`: `TJPARAM_LOSSLESS`.
    const TJPARAM_LOSSLESS: c_int = 15;

    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    // 16-bit samples are for *lossless* JPEG upstream, and a lossy 16-bit
    // compress is refused (`jcmaster.c:206`) before ownership matters — so
    // lossless is the only configuration in which this contract is observable
    // at all. P4-150 (#531) made the port agree on that refusal; the oracle
    // traces it as `compress16_lossy_roomy`.
    // SAFETY: live handle from `compressor`.
    unsafe {
        assert_eq!(tj3Set(handle, TJPARAM_LOSSLESS, 1), 0, "set lossless");
    }
    let src: Vec<u16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 65535) as u16)
        .collect();
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut buf: *mut u8 = original;
    let mut size: usize = ROOMY;

    // SAFETY: as above, with a 16-bit source.
    let rc: c_int = unsafe {
        tj3Compress16(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    assert_eq!(rc, 0, "tj3Compress16 must succeed");
    assert_kept("tj3Compress16", buf, original, size);

    // SAFETY: `original` came from `tj3Alloc` and is freed exactly once — the
    // point of the assertion above is that the library did not free it first.
    unsafe { tj3Free(original as *mut c_void) };
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
}

/// 4:4:4 packed YUV for `WIDTH x HEIGHT`: three full-resolution planes.
fn yuv_444_source() -> Vec<u8> {
    let plane: usize = WIDTH as usize * HEIGHT as usize;
    (0..plane * 3).map(|i| (i % 251) as u8).collect()
}

#[test]
fn tj3_compress_from_yuv8_keeps_the_callers_buffer() {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let src: Vec<u8> = yuv_444_source();
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut buf: *mut u8 = original;
    let mut size: usize = ROOMY;

    // SAFETY: `src` is a packed 4:4:4 image of the declared geometry with
    // align = 1; `buf`/`size` are a valid out-pair.
    let rc: c_int =
        unsafe { tj3CompressFromYUV8(handle, src.as_ptr(), WIDTH, 1, HEIGHT, &mut buf, &mut size) };
    assert_eq!(rc, 0, "tj3CompressFromYUV8 must succeed");
    assert_kept("tj3CompressFromYUV8", buf, original, size);

    // SAFETY: `original` came from `tj3Alloc` and is freed exactly once — the
    // point of the assertion above is that the library did not free it first.
    unsafe { tj3Free(original as *mut c_void) };
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
}

#[test]
fn tj3_compress_from_yuv_planes8_keeps_the_callers_buffer() {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let plane_len: usize = WIDTH as usize * HEIGHT as usize;
    let y: Vec<u8> = (0..plane_len).map(|i| (i % 251) as u8).collect();
    let cb: Vec<u8> = vec![128u8; plane_len];
    let cr: Vec<u8> = vec![128u8; plane_len];
    let planes: [*const u8; 3] = [y.as_ptr(), cb.as_ptr(), cr.as_ptr()];
    let strides: [c_int; 3] = [WIDTH, WIDTH, WIDTH];

    let original: *mut u8 = caller_buffer(ROOMY);
    let mut buf: *mut u8 = original;
    let mut size: usize = ROOMY;

    // SAFETY: three planes of `plane_len` bytes with matching strides, all
    // outliving the call; `buf`/`size` are a valid out-pair.
    let rc: c_int = unsafe {
        tj3CompressFromYUVPlanes8(
            handle,
            planes.as_ptr(),
            WIDTH,
            strides.as_ptr(),
            HEIGHT,
            &mut buf,
            &mut size,
        )
    };
    assert_eq!(rc, 0, "tj3CompressFromYUVPlanes8 must succeed");
    assert_kept("tj3CompressFromYUVPlanes8", buf, original, size);

    // SAFETY: `original` came from `tj3Alloc` and is freed exactly once — the
    // point of the assertion above is that the library did not free it first.
    unsafe { tj3Free(original as *mut c_void) };
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
}

/// A small baseline JPEG to transform.
fn source_jpeg() -> Vec<u8> {
    use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
    Encoder::new(
        &rgb_source(),
        WIDTH as usize,
        HEIGHT as usize,
        PixelFormat::Rgb,
    )
    .quality(80)
    .encode()
    .expect("encode source")
}

#[test]
fn tj3_transform_keeps_the_callers_destination_buffers() {
    let handle: *mut c_void = compressor(TJINIT_TRANSFORM);
    let jpeg: Vec<u8> = source_jpeg();

    let original: *mut u8 = caller_buffer(ROOMY);
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [ROOMY];
    // SAFETY: `TjTransform` is `#[repr(C)]` plain data; all-zero selects the
    // identity transform with no options, which is what this test wants.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: `jpeg` is a complete datastream; the three arrays each have the
    // one slot `n = 1` declares.
    let rc: c_int = unsafe {
        tj3Transform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
        )
    };
    assert_eq!(rc, 0, "tj3Transform must succeed");
    assert_kept("tj3Transform", dst_bufs[0], original, dst_sizes[0]);

    // SAFETY: `original` came from `tj3Alloc` and is freed exactly once — the
    // point of the assertion above is that the library did not free it first.
    unsafe { tj3Free(original as *mut c_void) };
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
}

// ---------------------------------------------------------------------------
// The refusal half: too small must be an error, not a silent resize.
// ---------------------------------------------------------------------------

/// A buffer that cannot hold the output must be *refused*, and left alone.
///
/// Reallocating instead would defeat the flag's whole purpose; overrunning it
/// is the heap overflow the capacity check exists to prevent.
#[test]
fn a_too_small_buffer_is_refused_rather_than_replaced() {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let src: Vec<u8> = rgb_source();
    let original: *mut u8 = caller_buffer(CRAMPED);
    let mut buf: *mut u8 = original;
    let mut size: usize = CRAMPED;

    // SAFETY: as in the success cases, but the declared capacity is far below
    // what the encode needs.
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    assert_eq!(rc, -1, "a too-small NOREALLOC buffer must be refused");
    assert_eq!(
        buf, original,
        "the refusal path must not swap the caller's pointer either"
    );

    // SAFETY: `original` came from `tj3Alloc` and is freed exactly once — the
    // point of the assertion above is that the library did not free it first.
    unsafe { tj3Free(original as *mut c_void) };
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
}

// ---------------------------------------------------------------------------
// C oracle: the contract compared against real TurboJPEG, not against itself.
// ---------------------------------------------------------------------------

mod helpers;

/// One case in the oracle's line format: `label rc kept produced`.
fn trace_line(label: &str, rc: c_int, slot: *mut u8, original: *mut u8, size: usize) -> String {
    format!(
        "{label} {rc} {} {}\n",
        c_int::from(slot == original),
        c_int::from(rc == 0 && size > 0)
    )
}

/// `tj3Compress8` with `capacity` bytes, or a NULL slot when `capacity` is 0.
fn compress8_case(label: &str, capacity: usize) -> String {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let src: Vec<u8> = rgb_source();
    let original: *mut u8 = if capacity == 0 {
        std::ptr::null_mut()
    } else {
        caller_buffer(capacity)
    };
    let mut buf: *mut u8 = original;
    let mut size: usize = capacity;

    // SAFETY: live handle; `src` holds the declared geometry; `buf`/`size` are
    // a valid out-pair whose buffer holds `capacity` bytes (or is NULL).
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, rc, buf, original, size);

    // SAFETY: free whatever we still own — the library swapped the pointer only
    // if it took the reallocating path, which under this flag it must not.
    unsafe {
        if buf != original && !buf.is_null() {
            tj3Free(buf as *mut c_void);
        }
        if !original.is_null() {
            tj3Free(original as *mut c_void);
        }
        tj3Destroy(handle);
    }
    line
}

/// The packed-YUV sibling, same shape.
fn yuv8_case(label: &str, capacity: usize) -> String {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let src: Vec<u8> = yuv_444_source();
    let original: *mut u8 = if capacity == 0 {
        std::ptr::null_mut()
    } else {
        caller_buffer(capacity)
    };
    let mut buf: *mut u8 = original;
    let mut size: usize = capacity;

    // SAFETY: as above, with a packed 4:4:4 source and align = 1.
    let rc: c_int =
        unsafe { tj3CompressFromYUV8(handle, src.as_ptr(), WIDTH, 1, HEIGHT, &mut buf, &mut size) };
    let line: String = trace_line(label, rc, buf, original, size);

    // SAFETY: as above.
    unsafe {
        if buf != original && !buf.is_null() {
            tj3Free(buf as *mut c_void);
        }
        if !original.is_null() {
            tj3Free(original as *mut c_void);
        }
        tj3Destroy(handle);
    }
    line
}

/// `tj3Compress12`, in the oracle's line format.
fn compress12_case(label: &str, capacity: usize) -> String {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let src: Vec<i16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 4096) as i16)
        .collect();
    let original: *mut u8 = if capacity == 0 {
        std::ptr::null_mut()
    } else {
        caller_buffer(capacity)
    };
    let mut buf: *mut u8 = original;
    let mut size: usize = capacity;

    // SAFETY: live handle; `src` holds the declared geometry; `buf`/`size` are
    // a valid out-pair (or a NULL slot, which is the case under test).
    let rc: c_int = unsafe {
        tj3Compress12(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, rc, buf, original, size);
    release(handle, buf, original);
    line
}

/// `tj3Compress16`. 16-bit samples are for *lossless* JPEG upstream — a lossy
/// 16-bit compress is refused before ownership matters (`jcmaster.c:206`) — so
/// lossless is the configuration the contract is exercised in. `lossless =
/// false` traces the refusal, which has an ownership contract of its own: the
/// caller's buffer must come back untouched.
fn compress16_case(label: &str, capacity: usize, lossless: bool) -> String {
    /// `turbojpeg.h`: `TJPARAM_LOSSLESS`.
    const TJPARAM_LOSSLESS: c_int = 15;

    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    if lossless {
        // SAFETY: live handle.
        unsafe {
            assert_eq!(tj3Set(handle, TJPARAM_LOSSLESS, 1), 0, "set lossless");
        }
    }
    let src: Vec<u16> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 65535) as u16)
        .collect();
    let original: *mut u8 = if capacity == 0 {
        std::ptr::null_mut()
    } else {
        caller_buffer(capacity)
    };
    let mut buf: *mut u8 = original;
    let mut size: usize = capacity;

    // SAFETY: as above, with 16-bit samples.
    let rc: c_int = unsafe {
        tj3Compress16(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, rc, buf, original, size);
    release(handle, buf, original);
    line
}

/// `tj3CompressFromYUVPlanes8`.
fn yuv_planes_case(label: &str, capacity: usize) -> String {
    let handle: *mut c_void = compressor(TJINIT_COMPRESS);
    let plane_len: usize = WIDTH as usize * HEIGHT as usize;
    let y: Vec<u8> = (0..plane_len).map(|i| (i % 251) as u8).collect();
    let cb: Vec<u8> = vec![128u8; plane_len];
    let cr: Vec<u8> = vec![128u8; plane_len];
    let planes: [*const u8; 3] = [y.as_ptr(), cb.as_ptr(), cr.as_ptr()];
    let strides: [c_int; 3] = [WIDTH, WIDTH, WIDTH];
    let original: *mut u8 = if capacity == 0 {
        std::ptr::null_mut()
    } else {
        caller_buffer(capacity)
    };
    let mut buf: *mut u8 = original;
    let mut size: usize = capacity;

    // SAFETY: three planes of `plane_len` bytes with matching strides, all
    // outliving the call.
    let rc: c_int = unsafe {
        tj3CompressFromYUVPlanes8(
            handle,
            planes.as_ptr(),
            WIDTH,
            strides.as_ptr(),
            HEIGHT,
            &mut buf,
            &mut size,
        )
    };
    let line: String = trace_line(label, rc, buf, original, size);
    release(handle, buf, original);
    line
}

/// `tj3Transform`, whose reusable slot is `dst_bufs[0]`.
fn transform_case(label: &str, capacity: usize) -> String {
    let handle: *mut c_void = compressor(TJINIT_TRANSFORM);
    let jpeg: Vec<u8> = source_jpeg();
    let original: *mut u8 = if capacity == 0 {
        std::ptr::null_mut()
    } else {
        caller_buffer(capacity)
    };
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [capacity];
    // SAFETY: `TjTransform` is `#[repr(C)]` plain data; all-zero is the
    // identity transform.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: `jpeg` is a complete datastream; each array has the one slot
    // `n = 1` declares.
    let rc: c_int = unsafe {
        tj3Transform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
        )
    };
    let line: String = trace_line(label, rc, dst_bufs[0], original, dst_sizes[0]);
    release(handle, dst_bufs[0], original);
    line
}

/// Free whatever is still ours and destroy the handle.
fn release(handle: *mut c_void, produced: *mut u8, original: *mut u8) {
    // SAFETY: `produced` differs from `original` only when the library took
    // the reallocating path, in which case it is the library's buffer to
    // return; `original` came from `tj3Alloc`.
    unsafe {
        if produced != original && !produced.is_null() {
            tj3Free(produced as *mut c_void);
        }
        if !original.is_null() {
            tj3Free(original as *mut c_void);
        }
        tj3Destroy(handle);
    }
}

/// A grayscale ICC-carrying source, compared on the **no-overrun invariant**.
///
/// Upstream succeeds here and this port refuses — on this path only. Upstream's
/// legacy-NOREALLOC capacity pre-read skips marker registration, so its
/// transform drops the profile (601 bytes); ours carries it (5619), exactly as
/// both libraries do on legacy `flags=0` and `tj3Transform` — P4-156 (#544), a
/// separate defect scoped to that ordering quirk. Tracing `rc` would fail for
/// that reason rather than for anything P4-151 governs.
///
/// What both must satisfy, and what sizing grayscale as 4:4:4 violates, is
/// narrower: never report success having written past the bound a compliant
/// caller allocated. That is what this line compares, so the cross-validation
/// is real without enshrining the divergence.
fn legacy_gray_no_overrun_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
    use libjpeg_turbo_rs_capi::{tjBufSize, tjTransform};

    const TJFLAG_NOREALLOC: c_int = 1024;
    /// `turbojpeg.h`: `TJSAMP_GRAY`.
    const TJSAMP_GRAY: c_int = 3;

    let gray_pixels: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize))
        .map(|i| (i % 251) as u8)
        .collect();
    let icc: Vec<u8> = vec![0x5Au8; 5_000];
    let jpeg: Vec<u8> = Encoder::new(
        &gray_pixels,
        WIDTH as usize,
        HEIGHT as usize,
        PixelFormat::Grayscale,
    )
    .quality(80)
    .icc_profile(&icc)
    .encode()
    .expect("encode grayscale source");

    let gray_bound: usize = tjBufSize(WIDTH, HEIGHT, TJSAMP_GRAY);
    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    // Over-allocated for the same reason as the standalone test: a port with
    // the bug must be caught reporting the overrun, not by performing it.
    let slack: usize = tjBufSize(WIDTH, HEIGHT, TJSAMP_444) + 4096;
    let original: *mut u8 = caller_buffer(gray_bound + slack);
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [0];
    // SAFETY: `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: each array has the one slot `n = 1` declares.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };
    let line: String = format!(
        "{label} {} {}\n",
        i32::from(dst_bufs[0] == original),
        i32::from(rc == 0 && dst_sizes[0] > gray_bound)
    );
    if dst_bufs[0] != original && !dst_bufs[0].is_null() {
        // SAFETY: library-allocated on a replacing path; freed once.
        unsafe { tj3Free(dst_bufs[0] as *mut c_void) };
    }
    // SAFETY: ours, freed once; destroyed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

/// A NULL source under the legacy wrapper, in the oracle's line format.
///
/// P4-151 review: the bridge sliced `jpeg_buf` before `tj3Transform` could
/// validate it. Comparing against C pins that the guard added for that agrees
/// with upstream's answer rather than merely avoiding the panic.
fn legacy_null_source_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::tjTransform;

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [0];
    // SAFETY: `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: the source pointer is deliberately NULL, which is the case under
    // test; the other arrays each have the one slot `n = 1` declares.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            std::ptr::null(),
            0,
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };
    let line: String = format!(
        "{label} {} {} {}\n",
        if rc == 0 { 0 } else { -1 },
        i32::from(dst_bufs[0] == original),
        i32::from(rc == 0 && dst_sizes[0] > 0)
    );
    // SAFETY: untouched, freed once; destroyed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

/// The **legacy** wrapper with `dstSizes[0] = 0`, in the oracle's line format.
///
/// P4-151: legacy `dstSizes` are outputs, so a caller that sized its buffer
/// with `tjTransformBufSize()` leaves them at zero. TJ3 reads the same slot as
/// a capacity, so before the bridge this failed with "buffer too small".
///
/// The destination is sized with `tjBufSize` on the source geometry — the
/// geometry-only bound, which is what upstream's wrapper computes and what the
/// capacity must not exceed.
fn legacy_transform_case(label: &str) -> String {
    use libjpeg_turbo_rs_capi::{tjBufSize, tjTransform};

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    let jpeg: Vec<u8> = source_jpeg();
    // SAFETY: `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    let bound: usize = tjBufSize(WIDTH, HEIGHT, TJSAMP_444);
    assert!(bound > 0, "tjBufSize must report a bound");
    let original: *mut u8 = caller_buffer(bound);
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [0];

    // SAFETY: each array has the one slot `n = 1` declares.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };
    let line: String = format!(
        "{label} {rc} {} {}\n",
        i32::from(dst_bufs[0] == original),
        i32::from(rc == 0 && dst_sizes[0] > 0)
    );
    if dst_bufs[0] != original && !dst_bufs[0].is_null() {
        // SAFETY: library-allocated on a replacing path; freed once.
        unsafe { tj3Free(dst_bufs[0] as *mut c_void) };
    }
    // SAFETY: ours, freed once; destroyed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
    line
}

/// The ownership contract, compared line for line against real TurboJPEG.
///
/// The tests above assert what *this* port does. That is necessary and not
/// sufficient: the first version of this fix allocated when the flag was set
/// and the slot was NULL, which every self-consistent assertion accepted.
/// Upstream refuses — the flag is a request *not to allocate*
/// (`jdatadst-tj.c:184-192`) — and this is what says so.
#[test]
fn norealloc_contract_matches_upstream_turbojpeg() {
    let Some(oracle) = helpers::build_oracle("norealloc_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found; the C oracle for \
             P4-145's ownership contract cannot be built. Set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };
    let c_trace: String = helpers::run_oracle(&oracle, &[]);

    let mut ours: String = String::new();
    for (label, capacity) in [("roomy", ROOMY), ("cramped", CRAMPED), ("null", 0)] {
        ours.push_str(&compress8_case(&format!("compress8_{label}"), capacity));
    }
    for (label, capacity) in [("roomy", ROOMY), ("cramped", CRAMPED), ("null", 0)] {
        ours.push_str(&yuv8_case(&format!("yuv8_{label}"), capacity));
    }
    // Every changed entry point, not the two that happened to be written
    // first: the NULL-slot divergence showed that a suite of self-consistency
    // assertions stays green while one call diverges.
    for (label, capacity) in [("roomy", ROOMY), ("cramped", CRAMPED), ("null", 0)] {
        ours.push_str(&compress12_case(&format!("compress12_{label}"), capacity));
    }
    for (label, capacity) in [("roomy", ROOMY), ("cramped", CRAMPED), ("null", 0)] {
        ours.push_str(&compress16_case(
            &format!("compress16_{label}"),
            capacity,
            true,
        ));
    }
    // Lossy 16-bit, which upstream refuses outright. This line needed the
    // lossless flag to agree until P4-150 (#531): the port accepted the
    // configuration and encoded a lossless stream anyway, so a trace taken here
    // would have disagreed for a reason unrelated to buffer ownership. That it
    // now agrees with no flag set is the proof the acceptance rule was fixed,
    // and it pins the refusal path's ownership behaviour too.
    ours.push_str(&compress16_case("compress16_lossy_roomy", ROOMY, false));
    for (label, capacity) in [("roomy", ROOMY), ("cramped", CRAMPED), ("null", 0)] {
        ours.push_str(&yuv_planes_case(&format!("yuvplanes_{label}"), capacity));
    }
    for (label, capacity) in [("roomy", ROOMY), ("cramped", CRAMPED), ("null", 0)] {
        ours.push_str(&transform_case(&format!("transform_{label}"), capacity));
    }
    // P4-151: the legacy wrapper's output-vs-capacity bridge.
    ours.push_str(&legacy_transform_case("legacy_transform_zero_size"));
    ours.push_str(&legacy_null_source_case("legacy_transform_null_source"));
    ours.push_str(&legacy_gray_no_overrun_case(
        "legacy_transform_gray_no_overrun",
    ));

    assert_eq!(
        ours, c_trace,
        "TJPARAM_NOREALLOC behaviour diverges from upstream TurboJPEG"
    );
}

// ---------------------------------------------------------------------------
// Two paths review found after the first version of the fix.
// ---------------------------------------------------------------------------

/// The legacy flag has to reach the TJ3 parameter.
///
/// `tjCompress2` takes `TJFLAG_NOREALLOC` and used to discard it, so
/// `tj3Compress8` saw the parameter unset and took the reallocating path. That
/// was survivable only while that path *leaked* the previous pointer; making it
/// `free()` — which is what upstream's `realloc` does, and what this change
/// adopted — turned it into an invalid free of caller-owned storage.
///
/// Upstream maps the flag in `processFlags` for every operation
/// (`turbojpeg.c:552`).
#[test]
fn legacy_tj_compress2_honours_tjflag_norealloc() {
    use libjpeg_turbo_rs_capi::tjCompress2;

    /// `turbojpeg.h:2793`.
    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let src: Vec<u8> = rgb_source();
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut buf: *mut u8 = original;
    let mut size: usize = ROOMY;

    // SAFETY: live handle; `src` holds the declared geometry; `buf`/`size` are
    // a valid out-pair over a `ROOMY`-byte buffer.
    let rc: c_int = unsafe {
        tjCompress2(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
            TJSAMP_444,
            80,
            TJFLAG_NOREALLOC,
        )
    };
    assert_eq!(rc, 0, "tjCompress2 must succeed");
    assert_eq!(
        buf, original,
        "TJFLAG_NOREALLOC must reach TJPARAM_NOREALLOC — otherwise the \
         caller's buffer is replaced and the original freed, which for a \
         legacy caller's own storage is an invalid free"
    );

    // SAFETY: still ours, freed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
}

/// `TJXOPT_NOOUTPUT` produces no output, so it needs no destination.
///
/// Upstream skips destination setup entirely for it (`turbojpeg.c:3007`), which
/// means a NULL slot is fine and a non-NULL slot is left alone. Requiring a
/// buffer here — as the first version of the NOREALLOC work did — rejects a
/// call upstream accepts.
#[test]
fn transform_with_no_output_needs_no_destination() {
    // `turbojpeg.h`: TJXOPT_* are bit flags; NOOUTPUT is bit 4.
    const TJXOPT_NOOUTPUT: c_int = 16;

    let handle: *mut c_void = compressor(TJINIT_TRANSFORM);
    let jpeg: Vec<u8> = source_jpeg();

    let mut dst_bufs: [*mut u8; 1] = [std::ptr::null_mut()];
    let mut dst_sizes: [usize; 1] = [0];
    // SAFETY: `TjTransform` is `#[repr(C)]` plain data; all-zero is the
    // identity transform, to which this test adds only the NOOUTPUT option.
    let mut transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];
    transforms[0].options = TJXOPT_NOOUTPUT;

    // SAFETY: `jpeg` is a complete datastream; the arrays each have the one
    // slot `n = 1` declares.
    let rc: c_int = unsafe {
        tj3Transform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
        )
    };
    assert_eq!(
        rc, 0,
        "TJXOPT_NOOUTPUT with a NULL destination must succeed — upstream never \
         sets up a destination for it"
    );
    assert!(
        dst_bufs[0].is_null(),
        "the slot must be left exactly as the caller passed it"
    );
    assert_eq!(dst_sizes[0], 0, "no output means no size");

    // SAFETY: destroyed once; nothing was allocated for us to free.
    unsafe { tj3Destroy(handle) };
}

/// The legacy size slot is an **output**, not a capacity.
///
/// `tjBufSize()`-sized buffers are the documented legacy idiom, and a caller
/// following it has no reason to write `*jpegSize` first. Forwarding the slot
/// to TJ3 — where the same field *is* an input capacity — turned that valid
/// call into "buffer too small". Upstream substitutes
/// `tj3JPEGBufSize(width, height, subsamp)` under the flag
/// (`turbojpeg.c:1282-1284`).
///
/// The distinguishing input is `size = 0`, which the previous test could not
/// catch because it passed a real capacity.
#[test]
fn legacy_tj_compress2_treats_the_size_slot_as_an_output() {
    use libjpeg_turbo_rs_capi::tjCompress2;

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let src: Vec<u8> = rgb_source();
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut buf: *mut u8 = original;
    // The legacy idiom: the caller sized its buffer with `tjBufSize()` and
    // never wrote this slot.
    let mut size: usize = 0;

    // SAFETY: live handle; `src` holds the declared geometry; `buf` is a
    // `ROOMY`-byte buffer and `size` is the legacy output slot.
    let rc: c_int = unsafe {
        tjCompress2(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
            TJSAMP_444,
            80,
            TJFLAG_NOREALLOC,
        )
    };
    assert_eq!(
        rc, 0,
        "a legacy NOREALLOC caller that left *jpegSize at 0 must succeed — the \
         slot is an output there, and upstream substitutes tj3JPEGBufSize"
    );
    assert_eq!(buf, original, "and the buffer pointer must survive");
    assert!(size > 0, "the slot must come back holding the real size");

    // SAFETY: still ours, freed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
}

/// The legacy transform wrapper maps the flag, and a nonzero size slot does
/// not change that.
///
/// Mapping `TJFLAG_NOREALLOC` is what stops the caller's destination buffers
/// being `free()`d, and P4-151 bridged the *size* semantics beside it: legacy
/// `dstSizes` are outputs, TJ3's are capacities, so a caller that sized its
/// destination with `tjTransformBufSize()` and left the slot at zero now gets a
/// transform rather than "buffer too small". A nonzero slot is *not* honoured
/// as a capacity — upstream replaces every slot with the geometry bound under
/// NOREALLOC (`turbojpeg.c:3122-3132`), and so does the bridge; this case pins
/// that a caller passing one is served identically.
#[test]
fn legacy_tj_transform_maps_the_flag_with_a_nonzero_size_slot() {
    use libjpeg_turbo_rs_capi::tjTransform;

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    let jpeg: Vec<u8> = source_jpeg();
    let original: *mut u8 = caller_buffer(ROOMY);

    // The slot is nonzero; the bridge replaces it with the geometry bound
    // (as upstream does) and the pointer survives.
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [ROOMY];
    // SAFETY: `TjTransform` is `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: `jpeg` is a complete datastream; each array has the one slot
    // `n = 1` declares.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };
    assert_eq!(rc, 0, "a nonzero size slot must not fail the call");
    assert_eq!(
        dst_bufs[0], original,
        "the flag must reach TJPARAM_NOREALLOC — otherwise the caller's \
         destination is replaced and the original freed"
    );

    // SAFETY: still ours, freed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
}

/// An invalid argument must not leave the handle's ownership behaviour changed.
///
/// Mapping `TJFLAG_NOREALLOC` before validating meant a call that returned -1
/// still altered `TJPARAM_NOREALLOC` — so the *next* call could free
/// caller-owned storage because of a call that failed. Upstream validates
/// first (`turbojpeg.c:1274-1280`).
#[test]
fn legacy_invalid_arguments_do_not_change_ownership_state() {
    use libjpeg_turbo_rs_capi::{tj3Get, tjCompress2};

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    // SAFETY: live handle.
    unsafe {
        assert_eq!(tj3Set(handle, TJPARAM_NOREALLOC, 0), 0, "start unset");
    }

    let src: Vec<u8> = rgb_source();
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;

    // Quality 500 is out of range, so this must fail — *without* having
    // enabled NOREALLOC on the way.
    // SAFETY: live handle; the buffers are valid for what they declare.
    let rc: c_int = unsafe {
        tjCompress2(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            &mut size,
            TJSAMP_444,
            500,
            TJFLAG_NOREALLOC,
        )
    };
    assert_eq!(rc, -1, "an out-of-range quality must be rejected");
    // SAFETY: live handle.
    assert_eq!(
        unsafe { tj3Get(handle, TJPARAM_NOREALLOC) },
        0,
        "a failed call must not leave TJPARAM_NOREALLOC set — the next call \
         would then treat a library-owned buffer as the caller's, or the \
         reverse"
    );

    // SAFETY: destroyed once; nothing was allocated.
    unsafe { tj3Destroy(handle) };
}

/// A NULL size pointer must be rejected, not papered over.
///
/// Routing the legacy call through a *local* `size_t` hid the NULL from
/// `tj3Compress8`, so a call upstream rejects returned success and allocated a
/// buffer the caller had no way to learn the size of — or to free.
#[test]
fn legacy_null_size_pointer_is_rejected() {
    use libjpeg_turbo_rs_capi::tjCompress2;

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let src: Vec<u8> = rgb_source();
    let mut buf: *mut u8 = std::ptr::null_mut();

    // SAFETY: live handle; `src` holds the declared geometry. The NULL size
    // pointer is the point of the test.
    let rc: c_int = unsafe {
        tjCompress2(
            handle,
            src.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut buf,
            std::ptr::null_mut(),
            TJSAMP_444,
            80,
            0,
        )
    };
    assert_eq!(rc, -1, "a NULL jpegSize must be rejected");
    assert!(
        buf.is_null(),
        "and nothing may be allocated for a caller that cannot be told the size"
    );

    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
}

// ---------------------------------------------------------------------------
// P4-151: the legacy `dstSizes` bridge.
// ---------------------------------------------------------------------------

/// A zero `dstSizes[i]` under `TJFLAG_NOREALLOC` must transform, not fail.
///
/// This is the convenience half of the legacy contract: `dstSizes` are
/// *outputs* there, so a caller that sized its buffer with
/// `tjTransformBufSize()` has no reason to fill them in. Upstream fills a
/// temporary capacity array from the transformed geometry
/// (`turbojpeg.c:3118-3132`); before P4-151 this port passed the zeros straight
/// through and TJ3 read them as capacities.
///
/// The destination is sized with `tj3TransformBufSize`, which is what a legacy
/// caller's `tjTransformBufSize()` maps to — so this also pins that the
/// capacity the bridge computes never exceeds the buffer that call describes.
#[test]
fn legacy_tj_transform_fills_a_zero_dst_size_from_geometry() {
    use libjpeg_turbo_rs_capi::{tj3DecompressHeader, tj3TransformBufSize, tjTransform};

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    let jpeg: Vec<u8> = source_jpeg();
    // SAFETY: `TjTransform` is `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // What a legacy caller sizes its destination with. Uses its own handle, so
    // the header parse here cannot be what makes the call under test work.
    let sizing_handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!sizing_handle.is_null(), "tj3Init for sizing");
    // SAFETY: live handle; `jpeg` is a complete datastream.
    let bound: usize = unsafe {
        assert_eq!(
            tj3DecompressHeader(sizing_handle, jpeg.as_ptr(), jpeg.len()),
            0,
            "header parse for sizing"
        );
        tj3TransformBufSize(sizing_handle, transforms.as_ptr())
    };
    assert!(bound > 0, "tj3TransformBufSize must report a bound");
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(sizing_handle) };

    let original: *mut u8 = caller_buffer(bound);
    let mut dst_bufs: [*mut u8; 1] = [original];
    // The whole point: left at zero, as the legacy contract permits.
    let mut dst_sizes: [usize; 1] = [0];

    // SAFETY: each array has the one slot `n = 1` declares; `original` is
    // `bound` bytes, which is what a legacy caller would have allocated.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };

    assert_eq!(
        rc, 0,
        "a zero dstSizes slot must be filled from the transformed geometry, \
         not passed through as a capacity of zero"
    );
    assert_eq!(
        dst_bufs[0], original,
        "the caller's destination must still be the one written"
    );
    assert!(
        dst_sizes[0] > 0 && dst_sizes[0] <= bound,
        "dstSizes must come back as the produced size ({}), within the bound \
         the caller sized against ({bound})",
        dst_sizes[0]
    );

    // SAFETY: still ours, freed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
}

/// The bridge must not parse the source header into the handle.
///
/// The rejected attempt used `tj3DecompressHeader` to learn the source
/// geometry. That writes shared state a later `tj3Compress*` on the same handle
/// would read, so the transform would silently reconfigure the caller's
/// compressor.
///
/// **Which parameter to watch was measured, not assumed.** The obvious guess —
/// that `TJPARAM_SUBSAMP` would be overwritten with the source's — is wrong in
/// this port: a header parse leaves it alone. What it *does* move, measured on
/// a 32x32 4:4:4 source with the handle pre-set, is `TJPARAM_JPEGHEIGHT`
/// (0 -> 32) and `TJPARAM_COLORSPACE` (-1 -> 1). An earlier version of this
/// test asserted `TJPARAM_SUBSAMP` and passed with the rejected approach
/// injected, which is how the mistake surfaced.
///
/// `TJPARAM_JPEGHEIGHT` is the sharper of the two: nothing but a header parse
/// sets it, so it staying at zero says the handle was never used to read the
/// source at all.
#[test]
fn legacy_tj_transform_does_not_parse_the_source_into_the_handle() {
    use libjpeg_turbo_rs_capi::{tj3Get, tjTransform};

    const TJFLAG_NOREALLOC: c_int = 1024;
    /// `turbojpeg.h`: `TJPARAM_JPEGHEIGHT`, set only by a header parse.
    const TJPARAM_JPEGHEIGHT: c_int = 6;
    /// `turbojpeg.h`: `TJPARAM_COLORSPACE`.
    const TJPARAM_COLORSPACE: c_int = 8;
    /// `turbojpeg.h`: `TJCS_YCbCr`, deliberately not what a parse would leave.
    const TJCS_YCBCR: c_int = 1;
    /// `turbojpeg.h`: `TJCS_GRAY`.
    const TJCS_GRAY: c_int = 2;

    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    // SAFETY: live handle.
    unsafe {
        assert_eq!(
            tj3Set(handle, TJPARAM_COLORSPACE, TJCS_GRAY),
            0,
            "pre-set a colour space the source parse would overwrite"
        );
        assert_eq!(
            tj3Get(handle, TJPARAM_JPEGHEIGHT),
            0,
            "no header has been parsed on this handle yet"
        );
    }

    let jpeg: Vec<u8> = source_jpeg();
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [0];
    // SAFETY: `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: each array has the one slot `n = 1` declares.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };
    assert_eq!(rc, 0, "the transform itself must succeed");

    // SAFETY: live handle.
    let (height_after, colorspace_after): (c_int, c_int) = unsafe {
        (
            tj3Get(handle, TJPARAM_JPEGHEIGHT),
            tj3Get(handle, TJPARAM_COLORSPACE),
        )
    };
    assert_eq!(
        height_after, 0,
        "TJPARAM_JPEGHEIGHT is set only by a header parse, so a non-zero value \
         means the size bridge parsed the source into the caller's handle"
    );
    assert_ne!(
        colorspace_after, TJCS_YCBCR,
        "the handle's colour space was set to TJCS_GRAY; a parse of this \
         4:4:4 source leaves TJCS_YCbCr, which a later tj3Compress* would use"
    );
    assert_eq!(
        colorspace_after, TJCS_GRAY,
        "the caller's colour space must survive the transform unchanged"
    );

    // SAFETY: still ours, freed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
}

/// A grayscale source must be sized as `TJSAMP_GRAY`, not as 4:4:4.
///
/// P4-151 review found this: `probe` reports `Subsampling::Unknown` for a
/// single-component image — there are no chroma planes to describe — and the
/// generic mapping turns `Unknown` into `TJSAMP_444`. A legacy caller allocates
/// `tjBufSize(w, h, TJSAMP_GRAY)`, which for 32x32 is 4096 bytes against 8192
/// for 4:4:4. Handing `tj3Transform` the larger figure as a *capacity* means
/// output between the two bounds is written past the end of the caller's
/// allocation.
///
/// The direction is what makes it a defect rather than waste: over-stating a
/// bound you allocate costs memory, over-stating a capacity you trust is an
/// overrun.
///
/// The destination here is deliberately sized at the grayscale bound — what a
/// compliant caller allocates — so a capacity computed as 4:4:4 is not merely
/// wrong on paper.
#[test]
fn legacy_tj_transform_sizes_a_grayscale_source_as_gray() {
    use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
    use libjpeg_turbo_rs_capi::{tjBufSize, tjTransform};

    const TJFLAG_NOREALLOC: c_int = 1024;
    /// `turbojpeg.h`: `TJSAMP_GRAY`.
    const TJSAMP_GRAY: c_int = 3;

    let gray_pixels: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize))
        .map(|i| (i % 251) as u8)
        .collect();
    // A profile large enough that the transformed output crosses the grayscale
    // bound but stays under the 4:4:4 one — the window where a capacity
    // computed as 4:4:4 lets `tj3Transform` write past the caller's buffer.
    // Without it the output of a 32x32 gray image is far below both bounds and
    // the test passes whether or not the gate is there; that is how the first
    // version of it failed to catch the defect.
    let icc: Vec<u8> = vec![0x5Au8; 5_000];
    let jpeg: Vec<u8> = Encoder::new(
        &gray_pixels,
        WIDTH as usize,
        HEIGHT as usize,
        PixelFormat::Grayscale,
    )
    .quality(80)
    .icc_profile(&icc)
    .encode()
    .expect("encode grayscale source");

    let gray_bound: usize = tjBufSize(WIDTH, HEIGHT, TJSAMP_GRAY);
    assert!(
        gray_bound < tjBufSize(WIDTH, HEIGHT, TJSAMP_444),
        "the grayscale bound must be the smaller of the two, or this test \
         cannot distinguish them"
    );

    // Over-allocate and canary, rather than allocating exactly `gray_bound`.
    // The bug this guards against makes `tj3Transform` write ~5.6 KiB into what
    // a compliant caller sizes at 4096, so a test that allocated exactly that
    // would *commit* the heap overflow whenever the regression returned —
    // corrupting the allocator, or aborting under a sanitizer, before its own
    // assertion could run. The extra room means the overrun is **observed**
    // instead: the canary past `gray_bound` records that bytes the caller never
    // owned were written.
    const CANARY: u8 = 0x5A;
    let slack: usize = tjBufSize(WIDTH, HEIGHT, TJSAMP_444) + 4096;
    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    let original: *mut u8 = caller_buffer(gray_bound + slack);
    // SAFETY: `original` is `gray_bound + slack` bytes from `tj3Alloc`.
    unsafe { std::ptr::write_bytes(original.add(gray_bound), CANARY, slack) };
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [0];
    // SAFETY: `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: each array has the one slot `n = 1` declares; `original` is at
    // least the bound a compliant legacy caller would have allocated.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };
    // The invariant, stated directly: never report success having written more
    // than the caller's buffer holds. Refusing is a correct outcome *for the
    // bytes we produce* — the profile makes the output genuinely larger than a
    // grayscale-sized destination. Upstream instead succeeds at 601 bytes on
    // this exact path: its legacy-NOREALLOC capacity pre-read skips marker
    // registration, so the profile (and every other marker) is dropped —
    // P4-156 (#544) tracks matching that quirk, and its criterion 5 re-pins
    // this test's mechanism once no marker can inflate the payload. What must
    // not happen on either side is `rc == 0` with a size past the bound, which
    // is the overrun a 4:4:4 capacity permits.
    assert!(
        !(rc == 0 && dst_sizes[0] > gray_bound),
        "reported success after producing {} bytes into a {gray_bound}-byte \
         buffer: the capacity handed to tj3Transform came from the 4:4:4 bound, \
         so everything past {gray_bound} was written outside the caller's \
         allocation",
        dst_sizes[0]
    );
    // The direct evidence, independent of the reported size: nothing beyond
    // what the caller allocated may have been touched.
    // SAFETY: the canary region is in bounds of the over-allocation above.
    let canary: &[u8] = unsafe { std::slice::from_raw_parts(original.add(gray_bound), slack) };
    let clobbered: Option<usize> = canary.iter().position(|byte| *byte != CANARY);
    assert!(
        clobbered.is_none(),
        "byte {} past the {gray_bound}-byte grayscale bound was overwritten — \
         a compliant caller allocates exactly that many, so this is a heap \
         overflow in its buffer",
        clobbered.unwrap_or(0)
    );

    // SAFETY: still ours, freed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
}

/// A NULL source under `TJFLAG_NOREALLOC` must return -1, not abort.
///
/// P4-151 review: the bridge sliced `jpeg_buf` before `tj3Transform` validated
/// it, and `from_raw_parts` requires a non-null pointer even for a zero-length
/// slice — a non-unwinding abort in debug, UB in release, where the API
/// documents -1.
#[test]
fn legacy_tj_transform_rejects_a_null_source_under_norealloc() {
    use libjpeg_turbo_rs_capi::tjTransform;

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    let original: *mut u8 = caller_buffer(ROOMY);
    let mut dst_bufs: [*mut u8; 1] = [original];
    let mut dst_sizes: [usize; 1] = [0];
    // SAFETY: `#[repr(C)]` plain data; all-zero is identity.
    let transforms: [TjTransform; 1] = [unsafe { std::mem::zeroed() }];

    // SAFETY: the source pointer is deliberately NULL, which is the case under
    // test; the remaining arrays each have the one slot `n = 1` declares.
    let rc: c_int = unsafe {
        tjTransform(
            handle,
            std::ptr::null(),
            0,
            1,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            transforms.as_ptr(),
            TJFLAG_NOREALLOC,
        )
    };
    assert_eq!(rc, -1, "a NULL source must be reported, not dereferenced");

    // SAFETY: untouched, freed once.
    unsafe {
        tj3Free(original as *mut c_void);
        tj3Destroy(handle);
    }
}
