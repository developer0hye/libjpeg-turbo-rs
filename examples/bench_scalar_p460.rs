//! P4-60 scalar measurement probe (issue #359). Run on a scalar-only
//! target (riscv64 under emulation) — prints in-process decode medians
//! plus a kernel-level A/B of the table-driven YCbCr→RGB against the
//! multiply form it replaced. Absolute times under emulation are
//! meaningless; ratios are the data.

use std::time::Instant;

fn median_us<F: FnMut()>(mut f: F, runs: usize) -> u128 {
    let mut samples: Vec<u128> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_micros());
    }
    samples.sort_unstable();
    samples[runs / 2]
}

/// The multiply form P4-60 replaced — kernel A/B oracle.
#[inline(always)]
fn multiply_pixel(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;
    let r = y + ((91881 * cr + 32768) >> 16);
    let g = y - ((22554 * cb + 46802 * cr + 32768) >> 16);
    let b = y + ((116130 * cb + 32768) >> 16);
    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("profile") {
        profile_stages();
        return;
    }
    let f640: &[u8] = include_bytes!("../tests/fixtures/photo_640x480_420.jpg");
    let f1080: &[u8] = include_bytes!("../tests/fixtures/photo_1920x1080_420.jpg");

    for (data, name, runs) in [
        (f640, "photo_640x480_420", 15),
        (f1080, "photo_1920x1080_420", 7),
    ] {
        let mut sink: usize = 0;
        let med = median_us(
            || {
                let img = libjpeg_turbo_rs::decompress(data).unwrap();
                sink ^= img.data.len();
            },
            runs,
        );
        println!("decode {name}: {med} us (sink {sink})");
    }

    // Kernel A/B: 1080 rows of 1920 px through each form.
    let y: Vec<u8> = (0..1920usize).map(|i| (i * 7) as u8).collect();
    let cb: Vec<u8> = (0..1920usize).map(|i| (i * 13) as u8).collect();
    let cr: Vec<u8> = (0..1920usize).map(|i| (i * 29) as u8).collect();
    let mut out: Vec<u8> = vec![0u8; 1920 * 3];

    let med_table = median_us(
        || {
            for _ in 0..1080 {
                libjpeg_turbo_rs::decode::color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut out, 1920);
                std::hint::black_box(&out);
            }
        },
        9,
    );
    let med_mul = median_us(
        || {
            for _ in 0..1080 {
                for x in 0..1920usize {
                    let (r, g, b) = multiply_pixel(y[x], cb[x], cr[x]);
                    out[x * 3] = r;
                    out[x * 3 + 1] = g;
                    out[x * 3 + 2] = b;
                }
                std::hint::black_box(&out);
            }
        },
        9,
    );
    println!("kernel ycbcr_row 1080p-frame: table={med_table} us multiply={med_mul} us");
}

// P4-60 per-stage profile (the acceptance criteria's outstanding step):
// micro-bench each scalar kernel at the exact volume a 1080p 4:2:0
// decode performs, so `decode_total - sum(stages)` bounds the residual
// (Huffman decode + plane plumbing). Invoked when the first CLI arg is
// `profile`; the plain run keeps its original output for comparability.
fn profile_stages() {
    use std::time::Instant;
    // 1080p 4:2:0: luma 240x135 = 32,400 blocks; chroma 2 x 8,100.
    let blocks_total: usize = 32_400 + 2 * 8_100;
    let coeffs: [i16; 64] = {
        let mut c = [0i16; 64];
        let mut i: usize = 0;
        while i < 64 {
            c[i] = ((i as i16 * 29) % 256) - 128;
            i += 1;
        }
        c
    };
    let t = Instant::now();
    let mut sink: i16 = 0;
    for _ in 0..blocks_total {
        let out = libjpeg_turbo_rs::decode::idct::idct_8x8(std::hint::black_box(&coeffs));
        sink ^= out[0];
    }
    println!(
        "stage idct_8x8 x{blocks_total}: {} us (sink {sink})",
        t.elapsed().as_micros()
    );

    // Chroma upsample: 2 components, 540 input rows -> 1080 output rows
    // of 960 -> 1920 samples (fancy h2v2 row form).
    let cur: Vec<u8> = (0..960usize).map(|i| (i * 3) as u8).collect();
    let nb: Vec<u8> = (0..960usize).map(|i| (i * 5) as u8).collect();
    let mut orow: Vec<u8> = vec![0u8; 1920];
    let t = Instant::now();
    for _ in 0..(2 * 1080usize) {
        libjpeg_turbo_rs::decode::upsample::fancy_h2v2_row(
            std::hint::black_box(&cur),
            std::hint::black_box(&nb),
            &mut orow,
            960,
        );
        std::hint::black_box(&orow);
    }
    println!("stage fancy_h2v2_row x2160: {} us", t.elapsed().as_micros());

    // Colour convert: 1080 rows of 1920 (post-table form).
    let y: Vec<u8> = (0..1920usize).map(|i| (i * 7) as u8).collect();
    let cb: Vec<u8> = (0..1920usize).map(|i| (i * 13) as u8).collect();
    let cr: Vec<u8> = (0..1920usize).map(|i| (i * 29) as u8).collect();
    let mut rgb: Vec<u8> = vec![0u8; 1920 * 3];
    let t = Instant::now();
    for _ in 0..1080 {
        libjpeg_turbo_rs::decode::color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, 1920);
        std::hint::black_box(&rgb);
    }
    println!("stage ycbcr_rows x1080: {} us", t.elapsed().as_micros());
}
