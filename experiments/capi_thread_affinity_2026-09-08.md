# P4-132 (#463): classic `cinfo` private state off `thread_local!` — 2026-09-08

**Question.** P4-16 set the bar for moving the per-`cinfo` side tables off
thread-local storage: a single-threaded `tj3Compress8` / `tj3Decompress8`
benchmark must stay within 1 % of the TLS-keyed baseline. P4-132 shipped the
move — not to a locked global map but onto the object itself, behind the
opaque `master` slot — so the question is whether anything got slower.

**Answer.** No. The TurboJPEG entry points share no code with the change and
sit inside noise; the classic decode loop, which *did* change (one field read
replaces a thread-local `HashMap` lookup on every entry point), got slightly
faster. Every delta below is inside the 1 % bar except the classic 64×64
improvement.

## Numbers

macOS aarch64 (Apple M-series, 16 GB), rustc 1.98.0, `--release` with
`lto = true`, `CARGO_BUILD_JOBS=4`, nothing else running. No fixed CPU
governor is available on this host, so the 640×480 medians drift by a few
percent between runs; the **minimum** per-iteration time over three
alternating runs (baseline, fix, baseline, fix, …) is the stable statistic
and is what the table reports. Medians per run are in the raw block below.

| Case | Baseline `ef4b061` (min µs/iter) | P4-132 (min µs/iter) | Delta |
|---|---:|---:|---:|
| `classic_decode_640x480` (create → destroy per iteration) | 1167.40 | 1163.92 | −0.30 % |
| `tj3Decompress8_640x480` | 1148.97 | 1145.09 | −0.34 % |
| `tj3Compress8_640x480` | 863.43 | 865.19 | +0.20 % |
| `classic_decode_64x64` (create → destroy per iteration) | 36.52 | 35.92 | −1.64 % |
| `tj3Decompress8_64x64` | 28.71 | 28.66 | −0.17 % |
| `tj3Compress8_64x64` | 13.46 | 13.37 | −0.67 % |

The classic case is the sensitive one: each iteration runs
`jpeg_CreateDecompress` → `jpeg_mem_src` → `jpeg_read_header` →
`jpeg_start_decompress` → 64 or 480 `jpeg_read_scanlines` →
`jpeg_finish_decompress` → `jpeg_destroy_decompress`, so every side-table
lookup the shim used to do is inside the timed region. At 64×64 the lookups
are a visible share of the work, which is where the improvement shows.

Raw output (three runs each; the first baseline and first fix run were taken
before the alternating pair):

```
BASE  classic_decode_640x480   min 1167.40  median 1182.32 | min 1251.72  median 1265.45 | min 1172.31  median 1200.34
AFTER classic_decode_640x480   min 1164.63  median 1174.59 | min 1163.92  median 1188.82 | min 1166.32  median 1211.98
BASE  tj3Decompress8_640x480   min 1148.97  median 1159.08 | min 1232.82  median 1246.23 | min 1164.43  median 1230.11
AFTER tj3Decompress8_640x480   min 1146.87  median 1151.55 | min 1145.09  median 1206.91 | min 1151.36  median 1159.57
BASE  tj3Compress8_640x480     min  863.43  median  868.34 | min  871.03  median  935.30 | min  872.20  median  876.78
AFTER tj3Compress8_640x480     min  865.19  median  872.33 | min  868.65  median  935.07 | min  868.34  median  938.35
BASE  classic_decode_64x64     min   36.52  median   36.76 | min   36.87  median   37.16 | min   36.68  median   37.03
AFTER classic_decode_64x64     min   35.92  median   36.07 | min   35.94  median   36.26 | min   36.02  median   36.20
BASE  tj3Decompress8_64x64     min   28.71  median   28.86 | min   28.80  median   28.97 | min   28.78  median   28.93
AFTER tj3Decompress8_64x64     min   28.83  median   29.24 | min   28.66  median   28.96 | min   28.77  median   28.98
BASE  tj3Compress8_64x64       min   13.49  median   13.54 | min   13.46  median   13.62 | min   13.47  median   13.58
AFTER tj3Compress8_64x64       min   13.38  median   13.50 | min   13.37  median   13.56 | min   13.46  median   13.57
```

## Method

A throwaway crate outside the workspace depends on `libjpeg-turbo-rs-capi`
by path — once at the baseline checkout, once at the fix — and links the
Rust entry points directly (the same functions the cdylib exports). The
fixture is a synthetic RGB gradient with per-pixel noise, encoded at quality
90, 4:2:0 by the crate's own encoder. Each case warms up, then times 9 rounds
of 200 (640×480) or 5000 (64×64) iterations and reports min/median/max
µs per iteration. Baseline and fix binaries were run one after the other,
never concurrently.

```toml
[package]
name = "p4132-bench"
version = "0.1.0"
edition = "2021"

[workspace]

[dependencies]
libjpeg-turbo-rs-capi = { path = "<checkout>/crates/libjpeg-turbo-rs-capi" }
libjpeg-turbo-rs = { path = "<checkout>" }

[profile.release]
lto = true
```

```rust
use std::ffi::{c_int, c_void};
use std::os::raw::c_ulong;
use std::time::Instant;

use libjpeg_turbo_rs::{PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_destroy_decompress, jpeg_finish_decompress, jpeg_mem_src,
    jpeg_read_header, jpeg_read_scanlines, jpeg_start_decompress, jpeg_std_error,
    JpegDecompressPublic, JpegErrorMgr,
};
use libjpeg_turbo_rs_capi::tj3::{tj3Destroy, tj3Init, tj3Set};
use libjpeg_turbo_rs_capi::{tj3Compress8, tj3Decompress8, tj3Free};

const TJINIT_COMPRESS: c_int = 0;
const TJINIT_DECOMPRESS: c_int = 1;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPF_RGB: c_int = 0;
const TJSAMP_420: c_int = 2;

fn pixels(w: usize, h: usize) -> Vec<u8> {
    let mut p = Vec::with_capacity(w * h * 3);
    let mut seed: u32 = 0x1234_5678;
    for y in 0..h {
        for x in 0..w {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 24) as u8 / 8;
            p.push(((x * 255) / w) as u8 ^ noise);
            p.push(((y * 255) / h) as u8 ^ noise);
            p.push((((x + y) * 127) / (w + h)) as u8 ^ noise);
        }
    }
    p
}

unsafe fn classic_decode(jpeg: &[u8], out: &mut [u8]) -> usize {
    let mut err: JpegErrorMgr = std::mem::zeroed();
    let mut cinfo: JpegDecompressPublic = std::mem::zeroed();
    cinfo.err = jpeg_std_error(&mut err);
    let cv = &mut cinfo as *mut JpegDecompressPublic as *mut c_void;
    jpeg_CreateDecompress(cv, 80, std::mem::size_of::<JpegDecompressPublic>());
    jpeg_mem_src(cv, jpeg.as_ptr(), jpeg.len() as c_ulong);
    assert_eq!(jpeg_read_header(cv, 1), 1);
    assert_eq!(jpeg_start_decompress(cv), 1);
    let stride = cinfo.output_width as usize * cinfo.output_components as usize;
    let h = cinfo.output_height as usize;
    let mut rows = 0usize;
    while (cinfo.output_scanline as usize) < h {
        let mut row = out.as_mut_ptr().add(cinfo.output_scanline as usize * stride);
        rows += jpeg_read_scanlines(cv, &mut row, 1) as usize;
    }
    assert_eq!(jpeg_finish_decompress(cv), 1);
    jpeg_destroy_decompress(cv);
    rows
}

fn measure<F: FnMut()>(name: &str, iters: usize, rounds: usize, mut f: F) {
    for _ in 0..(iters / 4).max(5) {
        f();
    }
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..rounds {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        samples.push(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{name}\tmin_us={:.2}\tmedian_us={:.2}\tmax_us={:.2}",
        samples[0],
        samples[samples.len() / 2],
        samples[samples.len() - 1]
    );
}

fn main() {
    for &(w, h) in &[(640usize, 480usize), (64, 64)] {
        let px = pixels(w, h);
        let jpeg =
            libjpeg_turbo_rs::compress(&px, w, h, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
        let mut out = vec![0u8; w * h * 3];
        let iters = if w == 640 { 200 } else { 5000 };
        measure(&format!("classic_decode_{w}x{h}"), iters, 9, || unsafe {
            assert_eq!(classic_decode(&jpeg, &mut out), h);
        });
        let dh = tj3Init(TJINIT_DECOMPRESS);
        measure(&format!("tj3Decompress8_{w}x{h}"), iters, 9, || unsafe {
            assert_eq!(
                tj3Decompress8(dh, jpeg.as_ptr(), jpeg.len(), out.as_mut_ptr(), 0, TJPF_RGB),
                0
            );
        });
        unsafe { tj3Destroy(dh) };
        let ch = tj3Init(TJINIT_COMPRESS);
        unsafe {
            assert_eq!(tj3Set(ch, TJPARAM_QUALITY, 90), 0);
            assert_eq!(tj3Set(ch, TJPARAM_SUBSAMP, TJSAMP_420), 0);
        }
        measure(&format!("tj3Compress8_{w}x{h}"), iters, 9, || unsafe {
            let mut buf: *mut u8 = std::ptr::null_mut();
            let mut size: usize = 0;
            assert_eq!(
                tj3Compress8(ch, px.as_ptr(), w as c_int, 0, h as c_int, TJPF_RGB, &mut buf, &mut size),
                0
            );
            tj3Free(buf as *mut c_void);
        });
        unsafe { tj3Destroy(ch) };
    }
}
```
