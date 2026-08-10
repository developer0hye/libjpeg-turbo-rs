//! `tj3Compress8` must honour `*jpeg_size` as the input capacity when
//! `TJPARAM_NOREALLOC` is set (issue #476 review; tracked as P4-145).
//!
//! Under NOREALLOC the caller owns the output buffer and `*jpeg_size` carries
//! its size on input. Upstream refuses to overrun it — `jdatadst-tj.c:92`
//! raises `JERR_BUFFER_SIZE` when the buffer fills and reallocation is
//! forbidden.
//!
//! We used to ignore the capacity entirely: the in-place path trusted that the
//! buffer was at least `tj3JPEGBufSize(...)` and `copy_nonoverlapping`'d the
//! encoded output regardless of what the caller declared. A caller doing
//! exactly what upstream permits — allocating a smaller buffer, declaring its
//! size, and relying on the library to refuse — got a heap overflow instead.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::{tj3Alloc, tj3Compress8, tj3Destroy, tj3Free, tj3Init, tj3Set};

const TJINIT_COMPRESS: c_int = 0;
const TJPARAM_NOREALLOC: c_int = 2;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJSAMP_444: c_int = 0;
const TJPF_RGB: c_int = 0;

const WIDTH: c_int = 64;
const HEIGHT: c_int = 64;

/// Deliberately far too small for a 64x64 q=95 4:4:4 JPEG, but a real
/// `tj3Alloc` allocation — so the only thing standing between the encoder and
/// the end of it is the capacity check.
const TINY_CAPACITY: usize = 64;

#[test]
fn tj3_compress8_norealloc_refuses_a_buffer_that_is_too_small() {
    let pixels: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    // SAFETY: `handle` is a live instance from `tj3Init`.
    unsafe {
        assert_eq!(tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444), 0);
        // High quality keeps the encoded output comfortably above TINY_CAPACITY
        // even for synthetic content.
        assert_eq!(tj3Set(handle, TJPARAM_QUALITY, 95), 0);
        assert_eq!(tj3Set(handle, TJPARAM_NOREALLOC, 1), 0);
    }

    let mut jpeg_buf: *mut u8 = tj3Alloc(TINY_CAPACITY) as *mut u8;
    assert!(!jpeg_buf.is_null(), "tj3Alloc");
    let mut jpeg_size: usize = TINY_CAPACITY;

    // SAFETY: `handle` is live; `pixels` is valid for the declared geometry;
    // `jpeg_buf` is a real `tj3Alloc` allocation of `TINY_CAPACITY` bytes and
    // `jpeg_size` declares exactly that. If the capacity check regresses, this
    // call writes several KiB into a 64-byte allocation and the sanitizer legs
    // report it.
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            pixels.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut jpeg_buf as *mut *mut u8,
            &mut jpeg_size as *mut usize,
        )
    };

    assert_eq!(
        rc, -1,
        "NOREALLOC with a {TINY_CAPACITY}-byte buffer must be refused, not overrun"
    );

    // SAFETY: both came from this library and are released exactly once.
    unsafe {
        tj3Free(jpeg_buf as *mut c_void);
        tj3Destroy(handle);
    }
}

/// The companion case: a buffer that *is* big enough must still succeed, and
/// must come back as the same pointer the caller supplied. Without this, the
/// capacity check above could be "fixed" by rejecting everything.
#[test]
fn tj3_compress8_norealloc_writes_in_place_when_the_buffer_fits() {
    let pixels: Vec<u8> = (0..(WIDTH as usize * HEIGHT as usize * 3))
        .map(|i| (i % 251) as u8)
        .collect();

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    // SAFETY: `handle` is a live instance from `tj3Init`.
    unsafe {
        assert_eq!(tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444), 0);
        assert_eq!(tj3Set(handle, TJPARAM_QUALITY, 95), 0);
        assert_eq!(tj3Set(handle, TJPARAM_NOREALLOC, 1), 0);
    }

    // Worst-case size for this geometry, with generous headroom.
    let capacity: usize = WIDTH as usize * HEIGHT as usize * 3 + 65536;
    let mut jpeg_buf: *mut u8 = tj3Alloc(capacity) as *mut u8;
    assert!(!jpeg_buf.is_null(), "tj3Alloc");
    let supplied: *mut u8 = jpeg_buf;
    let mut jpeg_size: usize = capacity;

    // SAFETY: as above, with a buffer sized for the worst case.
    let rc: c_int = unsafe {
        tj3Compress8(
            handle,
            pixels.as_ptr(),
            WIDTH,
            0,
            HEIGHT,
            TJPF_RGB,
            &mut jpeg_buf as *mut *mut u8,
            &mut jpeg_size as *mut usize,
        )
    };

    assert_eq!(rc, 0, "a sufficient NOREALLOC buffer must compress");
    assert_eq!(
        jpeg_buf, supplied,
        "NOREALLOC must not swap the caller's pointer"
    );
    assert!(jpeg_size > 0 && jpeg_size <= capacity);

    // SAFETY: both came from this library and are released exactly once.
    unsafe {
        tj3Free(jpeg_buf as *mut c_void);
        tj3Destroy(handle);
    }
}
