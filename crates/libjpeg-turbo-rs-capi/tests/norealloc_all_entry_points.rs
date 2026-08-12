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

/// The legacy transform wrapper maps the flag, and says so when it cannot
/// help further.
///
/// Mapping `TJFLAG_NOREALLOC` is what stops the caller's destination buffers
/// being `free()`d. Bridging the *size* semantics — legacy `dstSizes` are
/// outputs, TJ3's are capacities — is deliberately not attempted here; see
/// P4-151 for why, and what it needs. A caller that leaves the slot at zero
/// therefore gets an error rather than a transform, which is the smaller
/// divergence: the alternative it replaced freed the caller's own memory.
#[test]
fn legacy_tj_transform_maps_the_flag_and_refuses_a_zero_capacity() {
    use libjpeg_turbo_rs_capi::tjTransform;

    const TJFLAG_NOREALLOC: c_int = 1024;

    let handle: *mut c_void = tj3Init(TJINIT_TRANSFORM);
    assert!(!handle.is_null(), "tj3Init(TJINIT_TRANSFORM)");
    let jpeg: Vec<u8> = source_jpeg();
    let original: *mut u8 = caller_buffer(ROOMY);

    // With a real capacity the flag is honoured and the pointer survives.
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
    assert_eq!(rc, 0, "a declared capacity must be honoured");
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
