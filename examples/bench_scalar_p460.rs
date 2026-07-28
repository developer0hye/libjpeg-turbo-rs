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
