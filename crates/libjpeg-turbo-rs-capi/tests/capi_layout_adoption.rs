//! Behaviour pinned by the P4-139 `ImageLayout` adoption in this crate (#478).
//!
//! Centralising the span arithmetic was mostly a refactor, but three
//! caller-visible rules changed because the old hand-written chains were wrong
//! about them. Each gets a test here:
//!
//! * `tj3SaveImage8` read a **negative** `pitch` as "tight rows". Upstream
//!   rejects it outright (`turbojpeg-mp.c:511-513`).
//! * `tj3LoadImage8` ran `align.max(1)`, so `0`, a negative, and a
//!   non-power-of-two all silently became some stride the caller could not
//!   have predicted. Upstream refuses all three
//!   (`turbojpeg-mp.c:317-321`).
//! * The 12-/16-bit entry points guarded their source span with a bare
//!   `pitch.checked_mul(height)` over `i16`/`u16` elements, leaving the ×2 out
//!   of the chain. `slice::from_raw_parts` requires `len * size_of::<T>() <=
//!   isize::MAX`, so a geometry whose *sample* count fits `usize` while its
//!   *byte* span does not built an out-of-contract slice.
//!
//! `tj3Compress8`'s own source-span guard is pinned here too. It is older
//! than this change (P4-137 wrote it) but had no test, because the one file
//! that looked like its home said the arm was unreachable on 64-bit. It is
//! not: see `source_span_bounds::tj3_compress8_refuses_a_source_span_past_isize_max`.
//!
//! **Coverage, stated honestly.** The `isize::MAX` arm is reachable through
//! the public API only on the *compressing* side, where `width` and `height`
//! are both caller-supplied `c_int`s. On the decompressing side the height
//! comes from the JPEG's own SOF and is capped at 65535, so no legal stream
//! reaches the bound on a 64-bit host — that arm is 32-bit-only and is not
//! exercised anywhere. The compressing tests are `cfg`-gated to 64-bit
//! because on a 32-bit target the same call is refused one step earlier, by
//! the `usize` wrap, which proves a different arm.
//!
//! `tj3_decompress12_writes_only_the_pitched_extent` pins the span
//! *narrowing* rather than a bound, and its coverage is weaker than the
//! others': its assertions (rows match a dense decode, inter-row padding
//! untouched) hold under any runner, but the undefined behaviour it guards
//! against — a `from_raw_parts_mut` reaching past the caller's allocation —
//! is only *detected* under Miri or ASAN, and **no CI leg runs this file
//! under either today**. `sanitizers.yml` excludes integration tests and the
//! capi Miri step names `capi_create_abi_guards` alone. Adding a leg is not a
//! one-liner: the test performs a real encode and decode, so Miri stops on
//! the first SIMD intrinsic the dispatcher selects (measured: NEON
//! `llvm.aarch64.neon.ushl.v8i16`), which is why the library's own Miri run
//! passes `--skip simd::`. Getting a scalar-only capi build under Miri
//! belongs to P4-141 (#480), the soundness-verification program.
//!
//! No oracle process links stock TurboJPEG for `tj3LoadImage8` /
//! `tj3SaveImage8` (see `examples/` — the oracles cover compress, YUV and the
//! classic lifecycle), so the expected behaviour is cited to upstream source
//! lines rather than measured.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

use libjpeg_turbo_rs_capi::{
    tj3Destroy, tj3Free, tj3GetErrorStr, tj3Init, tj3LoadImage8, tj3SaveImage8, tj3Set,
};

const TJINIT_COMPRESS: c_int = 0;
const TJINIT_DECOMPRESS: c_int = 2;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJSAMP_GRAY: c_int = 3;
const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;

/// The handle's current error string, as the C caller would read it.
fn error_string(handle: *mut c_void) -> String {
    // SAFETY: `handle` is live; `tj3GetErrorStr` returns a NUL-terminated
    // buffer owned by the instance and valid until the next call on it.
    let raw: *const c_char = unsafe { tj3GetErrorStr(handle) };
    assert!(!raw.is_null(), "tj3GetErrorStr returned NULL");
    // SAFETY: as above.
    unsafe { CStr::from_ptr(raw) }
        .to_str()
        .expect("error string is not UTF-8")
        .to_string()
}

/// A `width x height` RGB ramp written as a binary PPM, plus the dense pixel
/// bytes it encodes. PPM keeps the loader's native format at `TJPF_RGB`, so
/// the bytes come back unswapped and can be compared directly.
fn write_rgb_ppm(dir: &Path, name: &str, width: usize, height: usize) -> (PathBuf, Vec<u8>) {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 17 + y) as u8);
            pixels.push((y * 23 + x) as u8);
            pixels.push((x * y + 7) as u8);
        }
    }
    let mut file: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
    file.extend_from_slice(&pixels);
    let path: PathBuf = dir.join(name);
    std::fs::write(&path, &file).expect("write PPM fixture");
    (path, pixels)
}

fn c_path(path: &Path) -> CString {
    CString::new(path.to_str().expect("path is UTF-8")).expect("path has no interior NUL")
}

// ---------------------------------------------------------------------------
// tj3SaveImage8: a negative pitch is an error, not "tight"
// ---------------------------------------------------------------------------

/// Upstream's argument gate refuses `pitch < 0` before it opens the output
/// file (`references/libjpeg-turbo/src/turbojpeg-mp.c:511-513`). This port
/// used `pitch <= 0` to mean "dense", so `-1` produced a file — built from a
/// stride the caller never asked for — and returned success.
#[test]
fn tj3_save_image8_rejects_a_negative_pitch() {
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let out: PathBuf = tmp.path().join("negative_pitch.ppm");
    let out_c: CString = c_path(&out);

    let width: usize = 8;
    let height: usize = 4;
    let pixels: Vec<u8> = vec![0x5Au8; width * height * 3];

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    // SAFETY: `handle` is live, the path is NUL-terminated, and `pixels` is a
    // valid allocation for the declared geometry.
    let rc: c_int = unsafe {
        tj3SaveImage8(
            handle,
            out_c.as_ptr(),
            pixels.as_ptr(),
            width as c_int,
            -1,
            height as c_int,
            TJPF_RGB,
        )
    };

    assert_eq!(
        rc, -1,
        "a negative pitch must be refused, not read as dense"
    );
    let message: String = error_string(handle);
    assert!(
        message.contains("pitch"),
        "the refusal must name the pitch; got {message:?}"
    );
    assert!(
        !out.exists(),
        "nothing may be written for a refused argument"
    );

    // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
    unsafe { tj3Destroy(handle) };
}

/// The companion: `pitch == 0` is upstream's own spelling of "dense"
/// (`turbojpeg-mp.c:587`) and must keep working, so the check above cannot be
/// "satisfied" by rejecting every non-positive pitch.
#[test]
fn tj3_save_image8_still_reads_pitch_zero_as_dense() {
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let out: PathBuf = tmp.path().join("dense.ppm");
    let out_c: CString = c_path(&out);

    let width: usize = 8;
    let height: usize = 4;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| i as u8).collect();

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    // SAFETY: as above.
    let rc: c_int = unsafe {
        tj3SaveImage8(
            handle,
            out_c.as_ptr(),
            pixels.as_ptr(),
            width as c_int,
            0,
            height as c_int,
            TJPF_RGB,
        )
    };
    assert_eq!(
        rc,
        0,
        "pitch 0 must still mean dense: {}",
        error_string(handle)
    );

    let written: Vec<u8> = std::fs::read(&out).expect("saved PPM must exist");
    assert!(
        written.starts_with(b"P6"),
        "a .ppm destination must produce a binary PPM"
    );
    assert!(
        written.ends_with(&pixels),
        "the saved pixels must be the caller's, byte for byte"
    );

    // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
    unsafe { tj3Destroy(handle) };
}

/// A genuinely padded pitch still reads the caller's rows at the right
/// offsets. This is what `ImageLayout::row_offset` replaced `y * stride`
/// with, and the source is allocated at exactly upstream's minimum —
/// `(height - 1) * pitch + width * bpp`, since upstream copies only
/// `width * ps` bytes from the final row (`turbojpeg-mp.c:594-598`) — so a
/// span that charged the last row's padding would read past this `Vec` and
/// the sanitizer legs would say so.
#[test]
fn tj3_save_image8_reads_a_padded_pitch_at_the_right_offsets() {
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let out: PathBuf = tmp.path().join("padded.ppm");
    let out_c: CString = c_path(&out);

    let width: usize = 5;
    let height: usize = 3;
    let row_bytes: usize = width * 3;
    let pitch: usize = 32;
    let mut padded: Vec<u8> = vec![0xEEu8; (height - 1) * pitch + row_bytes];
    let mut dense: Vec<u8> = Vec::with_capacity(row_bytes * height);
    for y in 0..height {
        for i in 0..row_bytes {
            let value: u8 = (y * row_bytes + i) as u8;
            padded[y * pitch + i] = value;
            dense.push(value);
        }
    }

    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    // SAFETY: `padded` is valid for the pitched extent declared here, which is
    // exactly the extent the entry point is documented to read.
    let rc: c_int = unsafe {
        tj3SaveImage8(
            handle,
            out_c.as_ptr(),
            padded.as_ptr(),
            width as c_int,
            pitch as c_int,
            height as c_int,
            TJPF_RGB,
        )
    };
    assert_eq!(rc, 0, "a padded pitch must save: {}", error_string(handle));

    let written: Vec<u8> = std::fs::read(&out).expect("saved PPM must exist");
    assert!(
        written.ends_with(&dense),
        "the padding must be dropped and only the pixels saved"
    );

    // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
    unsafe { tj3Destroy(handle) };
}

// ---------------------------------------------------------------------------
// tj3LoadImage8: `align` must be a positive power of two
// ---------------------------------------------------------------------------

/// Load the fixture with the given `align`, returning the raw result so each
/// test can assert its own outcome. The out-params are pre-set to sentinels so
/// "left unchanged on failure" is checkable.
fn load_with_align(handle: *mut c_void, path: &Path, align: c_int) -> (*mut u8, c_int, c_int) {
    let path_c: CString = c_path(path);
    let mut width: c_int = -7;
    let mut height: c_int = -7;
    let mut pixel_format: c_int = -1;
    // SAFETY: `handle` is live, the path is NUL-terminated, and all three
    // out-params are live locals of the right type.
    let buf: *mut u8 = unsafe {
        tj3LoadImage8(
            handle,
            path_c.as_ptr(),
            &mut width as *mut c_int,
            align,
            &mut height as *mut c_int,
            &mut pixel_format as *mut c_int,
        )
    };
    (buf, width, height)
}

/// `align < 1` is upstream's "Invalid argument"
/// (`references/libjpeg-turbo/src/turbojpeg-mp.c:317-319`). `align.max(1)`
/// used to turn it into a dense load that reported success.
#[test]
fn tj3_load_image8_rejects_a_non_positive_align() {
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let (path, _pixels) = write_rgb_ppm(tmp.path(), "align0.ppm", 5, 3);

    for align in [0, -4] {
        let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
        assert!(!handle.is_null(), "tj3Init");

        let (buf, width, height) = load_with_align(handle, &path, align);
        assert!(buf.is_null(), "align {align} must be refused");
        let message: String = error_string(handle);
        assert!(
            message.contains("align"),
            "the refusal must name align; got {message:?}"
        );
        assert_eq!(
            (width, height),
            (-7, -7),
            "out-params must be untouched on the refusal path"
        );

        // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
        unsafe { tj3Destroy(handle) };
    }
}

/// A positive non-power-of-two is upstream's "Alignment must be a power of 2"
/// (`turbojpeg-mp.c:320-321`), and is the case `align.max(1)` did not even
/// look at: `align == 3` produced rows padded to a multiple of 3, a stride no
/// documented caller computes. It is also the rule `tj3YUVBufSize` already
/// enforces for the same parameter (`bufsize.rs`), so accepting it here made
/// one library disagree with itself.
#[test]
fn tj3_load_image8_rejects_a_non_power_of_two_align() {
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let (path, _pixels) = write_rgb_ppm(tmp.path(), "align3.ppm", 5, 3);

    for align in [3, 6, 100] {
        let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
        assert!(!handle.is_null(), "tj3Init");

        let (buf, _width, _height) = load_with_align(handle, &path, align);
        assert!(buf.is_null(), "align {align} must be refused");
        let message: String = error_string(handle);
        assert!(
            message.contains("power of 2"),
            "the refusal must name the power-of-two rule; got {message:?}"
        );

        // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
        unsafe { tj3Destroy(handle) };
    }
}

/// The companion, and the check that the returned buffer is still the full
/// `pitch * height` rectangle upstream mallocs (`turbojpeg-mp.c:400-408`) —
/// padding past the final row included. `ImageLayout`'s strided total
/// deliberately excludes that last padding, so sizing this allocation from it
/// would hand back a buffer one pad short of what every caller's own
/// `pitch * height` arithmetic assumes. Reading all 48 bytes here is what
/// makes that shortfall observable under the sanitizer legs.
#[test]
fn tj3_load_image8_pads_rows_to_a_power_of_two_align() {
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let width: usize = 5;
    let height: usize = 3;
    let (path, pixels) = write_rgb_ppm(tmp.path(), "align4.ppm", width, height);

    let row_bytes: usize = width * 3;
    let stride: usize = 16; // 15 rounded up to a multiple of 4
    assert_eq!(row_bytes, 15, "fixture must have a non-aligned row");

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    let (buf, got_width, got_height) = load_with_align(handle, &path, 4);
    assert!(
        !buf.is_null(),
        "align 4 must load: {}",
        error_string(handle)
    );
    assert_eq!((got_width, got_height), (width as c_int, height as c_int));

    // SAFETY: the documented length of the returned buffer is `pitch *
    // height`; reading it in full is precisely the contract under test.
    let returned: Vec<u8> = unsafe { std::slice::from_raw_parts(buf, stride * height) }.to_vec();
    for y in 0..height {
        assert_eq!(
            &returned[y * stride..y * stride + row_bytes],
            &pixels[y * row_bytes..(y + 1) * row_bytes],
            "row {y} must be the file's pixels at the aligned offset"
        );
    }

    // SAFETY: the buffer came from `tj3LoadImage8`, which documents `tj3Free`
    // as its release path.
    unsafe { tj3Free(buf as *mut c_void) };
    // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
    unsafe { tj3Destroy(handle) };
}

// ---------------------------------------------------------------------------
// Source spans: the element size belongs inside the chain
// ---------------------------------------------------------------------------

/// Everything here is 64-bit-only, so it lives behind one gate rather than
/// three: on a 32-bit target the first factor already leaves `usize`, and a
/// refusal from that arm says nothing about the element size these tests are
/// about. Gating the tests alone would leave the constants unused and a
/// `-D warnings` 32-bit leg red.
#[cfg(target_pointer_width = "64")]
mod source_span_bounds {
    use super::*;

    use std::ffi::c_short;

    use libjpeg_turbo_rs_capi::{tj3Compress12, tj3Compress16, tj3Compress8};

    /// `TJPARAM_LOSSLESS`. Setting it short-circuits the quality/subsampling
    /// "must be specified" gate (P4-155), which otherwise refuses first — so
    /// without it a `-1` would prove only that a parameter was unset.
    const TJPARAM_LOSSLESS: c_int = 15;
    /// `TJPF_RGBA` — 4 bytes per pixel, the widest 8-bit format and so the
    /// one whose span crosses `isize::MAX` at the smallest dimensions.
    const TJPF_RGBA: c_int = 7;

    /// A buffer far too small for the declared geometry. Its only job is to
    /// be a valid, non-NULL address: the guard must reject before reading it.
    const TINY: [u8; 16] = [0u8; 16];

    /// The two geometries that break the 12-/16-bit guard's two arms, at
    /// `TJPF_RGB` (3 components) with `pitch == 0`, each verified against a
    /// `u128` model of the chain so it really lands where its label says:
    ///
    /// * `width = c_int::MAX, height = 800_000_000` — the byte span is
    ///   1.03e19: past `isize::MAX` (9.22e18) but inside `usize`, so this is
    ///   the **`isize::MAX` arm**. The old chain multiplied `pitch * height`
    ///   in *samples*, got 5.15e18, saw it fit `usize`, and built a
    ///   `from_raw_parts` slice covering 10.3 exabytes. This is the case the
    ///   x2 was missing from, and removing the `isize::MAX` bound makes
    ///   Rust's own `from_raw_parts` precondition check abort on it.
    /// * `width = c_int::MAX, height = c_int::MAX` — 2.77e19 bytes, so the
    ///   product leaves `usize` outright: the **wrap arm**.
    ///
    /// Both are refused with the same message, so the tests assert the
    /// refusal rather than which arm produced it; the pair exists so neither
    /// arm can rot unnoticed behind the other.
    ///
    /// For contrast, `c_int::MAX` square at *one* component totals
    /// 9.2233720e18 bytes and is legitimately **accepted** — just under the
    /// bound. The guard enforces `from_raw_parts`' precondition, not
    /// plausibility.
    const SPAN_OVERFLOW_GEOMETRIES: [(c_int, c_int, &str); 2] = [
        (c_int::MAX, 800_000_000, "isize::MAX arm"),
        (c_int::MAX, c_int::MAX, "usize wrap arm"),
    ];

    /// Shared body: declare `width x height` at `pixel_format` over `TINY`,
    /// and require the span guard — not a parameter gate — to refuse.
    fn refuses(
        arm: &str,
        width: c_int,
        height: c_int,
        pixel_format: c_int,
        call: impl FnOnce(*mut c_void, *mut *mut u8, *mut usize) -> c_int,
    ) {
        let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
        assert!(!handle.is_null(), "tj3Init");
        // SAFETY: `handle` is a live instance from `tj3Init`.
        assert_eq!(unsafe { tj3Set(handle, TJPARAM_LOSSLESS, 1) }, 0);

        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc: c_int = call(
            handle,
            &mut jpeg_buf as *mut *mut u8,
            &mut jpeg_size as *mut usize,
        );

        assert_eq!(
            rc, -1,
            "{arm}: {width}x{height} at TJPF {pixel_format} must be refused"
        );
        let message: String = error_string(handle);
        assert!(
            message.contains("overflow"),
            "{arm}: the span guard must be what refused, not a parameter gate; got {message:?}"
        );
        assert!(
            jpeg_buf.is_null(),
            "{arm}: no output buffer may be allocated on the rejection path"
        );

        // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
        unsafe { tj3Destroy(handle) };
    }

    /// The 12-bit source span must be measured in bytes, not samples.
    #[test]
    fn tj3_compress12_refuses_a_source_span_past_isize_max() {
        for (width, height, arm) in SPAN_OVERFLOW_GEOMETRIES {
            refuses(arm, width, height, TJPF_RGB, |handle, buf, size| {
                // SAFETY: `TINY` is a valid readable allocation for its own 16
                // bytes. The declared dimensions are the point of the test —
                // the guard must reject them before treating `TINY` as that
                // large.
                unsafe {
                    tj3Compress12(
                        handle,
                        TINY.as_ptr() as *const c_short,
                        width,
                        0,
                        height,
                        TJPF_RGB,
                        buf,
                        size,
                    )
                }
            });
        }
    }

    /// The 16-bit twin, over `*const u16`. Same arithmetic, same element
    /// size, same guard — pinned separately because the two entry points
    /// carry their own copies of the chain, and a fix applied to one and not
    /// the other is exactly how this class of bug survives.
    #[test]
    fn tj3_compress16_refuses_a_source_span_past_isize_max() {
        for (width, height, arm) in SPAN_OVERFLOW_GEOMETRIES {
            refuses(arm, width, height, TJPF_RGB, |handle, buf, size| {
                // SAFETY: as above.
                unsafe {
                    tj3Compress16(
                        handle,
                        TINY.as_ptr() as *const u16,
                        width,
                        0,
                        height,
                        TJPF_RGB,
                        buf,
                        size,
                    )
                }
            });
        }
    }

    /// `tj3Compress8`'s source-span guard, whose `isize::MAX` arm was
    /// believed 32-bit-only and is not: at `c_int::MAX` square in `TJPF_RGBA`
    /// the span is 18446744056529682436 bytes — inside `usize`
    /// (18446744073709551615) and well past `isize::MAX`
    /// (9223372036854775807). Nothing between the argument gate and the
    /// layout caps the dimensions, so a 64-bit caller reaches it.
    ///
    /// The guard itself predates this change (P4-137 added the
    /// `filter(|len| *len <= isize::MAX as usize)`); what was missing was any
    /// test, because `capi_span_overflow_guards.rs` covers the classic-ABI
    /// `JDIMENSION` rule and its module doc said this arm was unreachable
    /// here. It is cheap to pin, so it is pinned.
    #[test]
    fn tj3_compress8_refuses_a_source_span_past_isize_max() {
        refuses(
            "isize::MAX arm",
            c_int::MAX,
            c_int::MAX,
            TJPF_RGBA,
            |handle, buf, size| {
                // SAFETY: as above.
                unsafe {
                    tj3Compress8(
                        handle,
                        TINY.as_ptr(),
                        c_int::MAX,
                        0,
                        c_int::MAX,
                        TJPF_RGBA,
                        buf,
                        size,
                    )
                }
            },
        );
    }
}

/// A padded destination sized at exactly upstream's minimum —
/// `(height - 1) * pitch + width * components` samples, because upstream
/// writes only `width * ps` samples into the final row
/// (`turbojpeg-mp.c:242`). The old guard sized the slice at `pitch * height`,
/// so `from_raw_parts_mut` covered memory past this `Vec` before a single
/// sample was written: undefined behaviour on its own terms, and what the
/// sanitizer legs catch.
///
/// The content assertions keep the test meaningful without a sanitizer, and
/// are exact rather than tolerant because the reference is *the same JPEG
/// decoded densely*, not the pre-compression samples: pitch may not change a
/// single decoded value, whatever the quantiser did to them. (12-bit lossless
/// would allow comparing against the source directly, but `decompress_12bit`
/// does not read SOF3 — a separate gap, not this one.)
#[test]
fn tj3_decompress12_writes_only_the_pitched_extent() {
    // Scoped so the file-level imports stay to what every pointer width uses;
    // the 12-/16-bit compress entry points are otherwise only reached from
    // the `source_span_bounds` module, which is 64-bit-only.
    use libjpeg_turbo_rs_capi::{tj3Compress12, tj3Decompress12};

    let width: usize = 9;
    let height: usize = 4;
    let mut src: Vec<i16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            src.push(((y * width + x) * 37 % 4096) as i16);
        }
    }

    let encoder: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!encoder.is_null(), "tj3Init");
    // A lossy compress refuses an unset quality or subsampling (P4-155).
    // SAFETY: `encoder` is a live instance from `tj3Init`.
    assert_eq!(unsafe { tj3Set(encoder, TJPARAM_QUALITY, 95) }, 0);
    // SAFETY: as above.
    assert_eq!(unsafe { tj3Set(encoder, TJPARAM_SUBSAMP, TJSAMP_GRAY) }, 0);

    let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
    let mut jpeg_size: usize = 0;
    // SAFETY: `src` holds exactly `width * height` samples, which is what a
    // pitch of 0 and TJPF_GRAY declare.
    let rc: c_int = unsafe {
        tj3Compress12(
            encoder,
            src.as_ptr(),
            width as c_int,
            0,
            height as c_int,
            TJPF_GRAY,
            &mut jpeg_buf as *mut *mut u8,
            &mut jpeg_size as *mut usize,
        )
    };
    assert_eq!(rc, 0, "12-bit compress: {}", error_string(encoder));
    assert!(!jpeg_buf.is_null() && jpeg_size > 4);

    // Reference: the same stream decoded with no pitch at all.
    let decoder: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!decoder.is_null(), "tj3Init");
    let mut dense: Vec<i16> = vec![0i16; width * height];
    // SAFETY: `jpeg_buf`/`jpeg_size` came from the compress above; `dense`
    // holds the `width * height` samples a pitch of 0 declares.
    let rc: c_int = unsafe {
        tj3Decompress12(
            decoder,
            jpeg_buf,
            jpeg_size,
            dense.as_mut_ptr(),
            0,
            TJPF_GRAY,
        )
    };
    assert_eq!(rc, 0, "dense 12-bit decompress: {}", error_string(decoder));

    let pitch: usize = width + 7;
    let mut padded: Vec<i16> = vec![-1i16; (height - 1) * pitch + width];
    // SAFETY: `padded` is allocated at exactly the pitched extent the call is
    // allowed to touch — the point of the test.
    let rc: c_int = unsafe {
        tj3Decompress12(
            decoder,
            jpeg_buf,
            jpeg_size,
            padded.as_mut_ptr(),
            pitch as c_int,
            TJPF_GRAY,
        )
    };
    assert_eq!(rc, 0, "padded 12-bit decompress: {}", error_string(decoder));

    for y in 0..height {
        assert_eq!(
            &padded[y * pitch..y * pitch + width],
            &dense[y * width..(y + 1) * width],
            "row {y} must hold the same samples the dense decode produced"
        );
        // The padding between rows belongs to the caller and must be
        // untouched. The last row has none, which is the whole point.
        if y + 1 < height {
            assert!(
                padded[y * pitch + width..(y + 1) * pitch]
                    .iter()
                    .all(|&v| v == -1),
                "row {y}'s padding must be left alone"
            );
        }
    }

    // SAFETY: `jpeg_buf` came from `tj3Compress12`, which documents `tj3Free`.
    unsafe { tj3Free(jpeg_buf as *mut c_void) };
    // SAFETY: both handles came from `tj3Init` and are destroyed exactly once.
    unsafe { tj3Destroy(encoder) };
    // SAFETY: as above.
    unsafe { tj3Destroy(decoder) };
}
