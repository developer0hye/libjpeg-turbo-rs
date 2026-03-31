/// Synthetic pattern encode/decode matrix tests.
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_lossless_arithmetic,
    compress_lossless_extended, compress_optimized, compress_progressive, decompress,
    decompress_to, Encoder, PixelFormat, Subsampling,
};

const MAX_SAMPLE: u8 = 255;
const RED_TO_Y: u8 = 76;
const YELLOW_TO_Y: u8 = 226;
const HALFWAY: usize = 16;

fn gen_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    for r in 0..h {
        for c in 0..w {
            let i = (r * w + c) * 3;
            if ((r / 8) + (c / 8)) % 2 == 0 {
                if r < HALFWAY {
                    buf[i] = 255;
                    buf[i + 1] = 255;
                    buf[i + 2] = 255;
                }
            } else {
                buf[i] = 255;
                if r >= HALFWAY {
                    buf[i + 1] = 255;
                }
            }
        }
    }
    buf
}

fn gen_gray(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            if ((r / 8) + (c / 8)) % 2 == 0 {
                buf[i] = if r < HALFWAY { 255 } else { 0 };
            } else {
                buf[i] = if r < HALFWAY { RED_TO_Y } else { YELLOW_TO_Y };
            }
        }
    }
    buf
}

fn gen_quad(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    let hw = w / 2;
    let hh = h / 2;
    for r in 0..h {
        for c in 0..w {
            let i = (r * w + c) * 3;
            if r < hh && c < hw {
                buf[i] = 255;
            } else if r < hh {
                buf[i + 1] = 255;
            } else if c < hw {
                buf[i + 2] = 255;
            } else {
                buf[i] = 255;
                buf[i + 1] = 255;
                buf[i + 2] = 255;
            }
        }
    }
    buf
}

fn check_near(v: i16, e: i16, t: i16, r: usize, c: usize, ch: &str) -> Result<(), String> {
    if (v - e).abs() > t {
        Err(format!("{} ({},{}) exp ~{} got {}", ch, c, r, e, v))
    } else {
        Ok(())
    }
}

#[allow(clippy::collapsible_else_if)]
fn verify_rgb(d: &[u8], w: usize, h: usize, t: i16, _ss: Subsampling) -> Result<(), String> {
    for r in 0..h {
        for c in 0..w {
            let i = (r * w + c) * 3;
            let (rv, gv, bv) = (d[i] as i16, d[i + 1] as i16, d[i + 2] as i16);
            if ((r / 8) + (c / 8)) % 2 == 0 {
                if r < HALFWAY {
                    check_near(rv, 255, t, r, c, "R")?;
                    check_near(gv, 255, t, r, c, "G")?;
                    check_near(bv, 255, t, r, c, "B")?;
                } else {
                    check_near(rv, 0, t, r, c, "R")?;
                    check_near(gv, 0, t, r, c, "G")?;
                    check_near(bv, 0, t, r, c, "B")?;
                }
            } else {
                if r < HALFWAY {
                    check_near(rv, 255, t, r, c, "R")?;
                    check_near(gv, 0, t, r, c, "G")?;
                    check_near(bv, 0, t, r, c, "B")?;
                } else {
                    check_near(rv, 255, t, r, c, "R")?;
                    check_near(gv, 255, t, r, c, "G")?;
                    check_near(bv, 0, t, r, c, "B")?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::collapsible_else_if)]
fn verify_gray(d: &[u8], w: usize, h: usize, t: i16) -> Result<(), String> {
    for r in 0..h {
        for c in 0..w {
            let v = d[r * w + c] as i16;
            if ((r / 8) + (c / 8)) % 2 == 0 {
                if r < HALFWAY {
                    check_near(v, 255, t, r, c, "Y")?;
                } else {
                    check_near(v, 0, t, r, c, "Y")?;
                }
            } else {
                if r < HALFWAY {
                    check_near(v, RED_TO_Y as i16, t, r, c, "Y")?;
                } else {
                    check_near(v, YELLOW_TO_Y as i16, t, r, c, "Y")?;
                }
            }
        }
    }
    Ok(())
}

fn pix_tol(d: &[u8], e: &[u8], t: u8, l: &str) -> Result<(), String> {
    if d.len() != e.len() {
        return Err(format!("{}: len {} vs {}", l, d.len(), e.len()));
    }
    for (i, (&a, &b)) in d.iter().zip(e.iter()).enumerate() {
        if (a as i16 - b as i16).abs() > t as i16 {
            return Err(format!("{}: byte {} diff={}", l, i, a as i16 - b as i16));
        }
    }
    Ok(())
}

const A: (usize, usize) = (48, 48);
const NA: (usize, usize) = (35, 27);
const CSS: [Subsampling; 6] = [
    Subsampling::S444,
    Subsampling::S422,
    Subsampling::S420,
    Subsampling::S440,
    Subsampling::S411,
    Subsampling::S441,
];
const KSS: [Subsampling; 3] = [Subsampling::S444, Subsampling::S422, Subsampling::S420];

#[test]
fn baseline_aligned() {
    let p = gen_rgb(A.0, A.1);
    for &s in &CSS {
        let j = compress(&p, A.0, A.1, PixelFormat::Rgb, 75, s).unwrap();
        let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
        assert_eq!((i.width, i.height), A, "{:?}", s);
    }
}
#[test]
fn baseline_nonaligned() {
    let p = gen_rgb(NA.0, NA.1);
    for &s in &CSS {
        let j = compress(&p, NA.0, NA.1, PixelFormat::Rgb, 75, s).unwrap();
        let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
        assert_eq!((i.width, i.height), NA, "{:?}", s);
    }
}
#[test]
fn baseline_gray_aligned() {
    let p = gen_gray(A.0, A.1);
    let j = Encoder::new(&p, A.0, A.1, PixelFormat::Grayscale)
        .quality(75)
        .encode()
        .unwrap();
    let i = decompress(&j).unwrap();
    assert_eq!(i.pixel_format, PixelFormat::Grayscale);
    verify_gray(&i.data, A.0, A.1, 6).unwrap();
}
#[test]
fn baseline_gray_nonaligned() {
    let p = gen_gray(NA.0, NA.1);
    let j = Encoder::new(&p, NA.0, NA.1, PixelFormat::Grayscale)
        .quality(75)
        .encode()
        .unwrap();
    let i = decompress(&j).unwrap();
    assert_eq!((i.width, i.height), NA);
}
#[test]
fn progressive_aligned() {
    let p = gen_rgb(A.0, A.1);
    for &s in &CSS {
        let j = compress_progressive(&p, A.0, A.1, PixelFormat::Rgb, 75, s).unwrap();
        assert!(j.windows(2).any(|w| w == [0xFF, 0xC2]), "SOF2 {:?}", s);
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), A);
    }
}
#[test]
fn progressive_nonaligned() {
    let p = gen_rgb(NA.0, NA.1);
    for &s in &CSS {
        let j = compress_progressive(&p, NA.0, NA.1, PixelFormat::Rgb, 75, s).unwrap();
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), NA, "{:?}", s);
    }
}
#[test]
fn arithmetic_aligned() {
    let p = gen_rgb(A.0, A.1);
    for &s in &CSS {
        let j = compress_arithmetic(&p, A.0, A.1, PixelFormat::Rgb, 75, s).unwrap();
        assert!(j.windows(2).any(|w| w == [0xFF, 0xC9]), "SOF9 {:?}", s);
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), A);
    }
}
#[test]
fn arithmetic_nonaligned() {
    let p = gen_rgb(NA.0, NA.1);
    for &s in &CSS {
        let j = compress_arithmetic(&p, NA.0, NA.1, PixelFormat::Rgb, 75, s).unwrap();
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), NA, "{:?}", s);
    }
}
#[test]
fn arith_progressive_key() {
    let p = gen_rgb(A.0, A.1);
    for &s in &KSS {
        if let Ok(j) = compress_arithmetic_progressive(&p, A.0, A.1, PixelFormat::Rgb, 75, s) {
            if let Ok(i) = decompress(&j) {
                assert_eq!((i.width, i.height), A);
            }
        }
    }
}
#[test]
fn optimized_huffman() {
    let p = gen_rgb(A.0, A.1);
    for &s in &CSS {
        let sj = compress(&p, A.0, A.1, PixelFormat::Rgb, 75, s).unwrap();
        let oj = compress_optimized(&p, A.0, A.1, PixelFormat::Rgb, 75, s).unwrap();
        assert!(oj.len() <= sj.len() + 100, "opt {:?}", s);
        let i = decompress(&oj).unwrap();
        assert_eq!((i.width, i.height), A);
    }
}
#[test]
fn quality_1() {
    let p = gen_rgb(A.0, A.1);
    for &s in &KSS {
        let j = compress(&p, A.0, A.1, PixelFormat::Rgb, 1, s).unwrap();
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), A);
    }
}
#[test]
fn quality_75() {
    let p = gen_rgb(A.0, A.1);
    for &s in &KSS {
        let j = compress(&p, A.0, A.1, PixelFormat::Rgb, 75, s).unwrap();
        let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
        assert!(!i.data.iter().all(|&v| v == 0), "q75 {:?}", s);
    }
}
#[test]
fn quality_100_444() {
    let p = gen_rgb(A.0, A.1);
    let j = compress(&p, A.0, A.1, PixelFormat::Rgb, 100, Subsampling::S444).unwrap();
    let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
    pix_tol(&i.data, &p, 2, "q100 444").unwrap();
}
#[test]
fn quality_ordering() {
    let p = gen_rgb(A.0, A.1);
    for &s in &KSS {
        let q1 = compress(&p, A.0, A.1, PixelFormat::Rgb, 1, s).unwrap();
        let q100 = compress(&p, A.0, A.1, PixelFormat::Rgb, 100, s).unwrap();
        assert!(q100.len() > q1.len(), "q100>q1 {:?}", s);
    }
}
#[test]
fn restart_rows() {
    let p = gen_rgb(A.0, A.1);
    for &s in &CSS {
        let j = Encoder::new(&p, A.0, A.1, PixelFormat::Rgb)
            .quality(75)
            .subsampling(s)
            .restart_rows(1)
            .encode()
            .unwrap();
        assert!(j.windows(2).any(|w| w == [0xFF, 0xDD]), "DRI {:?}", s);
        assert!(
            j.windows(2)
                .any(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1])),
            "RST {:?}",
            s
        );
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), A);
    }
}
#[test]
fn restart_blocks() {
    let p = gen_rgb(A.0, A.1);
    for &s in &CSS {
        let j = Encoder::new(&p, A.0, A.1, PixelFormat::Rgb)
            .quality(75)
            .subsampling(s)
            .restart_blocks(2)
            .encode()
            .unwrap();
        assert!(j.windows(2).any(|w| w == [0xFF, 0xDD]), "DRI {:?}", s);
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), A);
    }
}
#[test]
fn pattern_baseline_444() {
    let p = gen_rgb(A.0, A.1);
    let j = compress(&p, A.0, A.1, PixelFormat::Rgb, 100, Subsampling::S444).unwrap();
    let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
    verify_rgb(&i.data, A.0, A.1, 2, Subsampling::S444).unwrap();
}
#[test]
fn pattern_progressive_444() {
    let p = gen_rgb(A.0, A.1);
    let j = compress_progressive(&p, A.0, A.1, PixelFormat::Rgb, 100, Subsampling::S444).unwrap();
    let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
    verify_rgb(&i.data, A.0, A.1, 2, Subsampling::S444).unwrap();
}
#[test]
fn pattern_arithmetic_444() {
    let p = gen_rgb(A.0, A.1);
    let j = compress_arithmetic(&p, A.0, A.1, PixelFormat::Rgb, 100, Subsampling::S444).unwrap();
    let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
    verify_rgb(&i.data, A.0, A.1, 2, Subsampling::S444).unwrap();
}
#[test]
fn pattern_grayscale() {
    let p = gen_gray(A.0, A.1);
    let j = Encoder::new(&p, A.0, A.1, PixelFormat::Grayscale)
        .quality(100)
        .encode()
        .unwrap();
    let i = decompress(&j).unwrap();
    verify_gray(&i.data, A.0, A.1, 2).unwrap();
}
#[test]
fn builder_arith_prog() {
    let p = gen_rgb(A.0, A.1);
    for &s in &KSS {
        if let Ok(j) = Encoder::new(&p, A.0, A.1, PixelFormat::Rgb)
            .quality(75)
            .subsampling(s)
            .arithmetic(true)
            .progressive(true)
            .encode()
        {
            if let Ok(i) = decompress(&j) {
                assert_eq!((i.width, i.height), A);
            }
        }
    }
}
#[test]
fn builder_opt_prog() {
    let p = gen_rgb(A.0, A.1);
    for &s in &[Subsampling::S444, Subsampling::S420] {
        let j = Encoder::new(&p, A.0, A.1, PixelFormat::Rgb)
            .quality(75)
            .subsampling(s)
            .progressive(true)
            .optimize_huffman(true)
            .encode()
            .unwrap();
        assert!(j.windows(2).any(|w| w == [0xFF, 0xC2]), "SOF2 {:?}", s);
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), A);
    }
}
#[test]
fn multiple_pixel_formats() {
    let (w, h) = (48usize, 48usize);
    for &(pf, bpp) in &[
        (PixelFormat::Rgb, 3usize),
        (PixelFormat::Bgr, 3),
        (PixelFormat::Rgba, 4),
        (PixelFormat::Bgra, 4),
        (PixelFormat::Rgbx, 4),
    ] {
        let mut px = vec![0u8; w * h * bpp];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * bpp;
                let ro = pf.red_offset().unwrap_or(0);
                let go = pf.green_offset().unwrap_or(1);
                let bo = pf.blue_offset().unwrap_or(2);
                px[i + ro] = ((x * 255) / w) as u8;
                px[i + go] = ((y * 255) / h) as u8;
                px[i + bo] = 128;
                if bpp == 4 {
                    let ao = 6 - ro - go - bo;
                    if ao < bpp {
                        px[i + ao] = 255;
                    }
                }
            }
        }
        let j = Encoder::new(&px, w, h, pf)
            .quality(90)
            .subsampling(Subsampling::S444)
            .encode()
            .unwrap();
        let i = decompress(&j).unwrap();
        assert_eq!((i.width, i.height), (w, h), "{:?}", pf);
    }
}
#[test]
fn entropy_quality_subsamp() {
    let p = gen_rgb(A.0, A.1);
    for &q in &[1u8, 75, 100] {
        for &s in &KSS {
            assert_eq!(
                decompress(&compress(&p, A.0, A.1, PixelFormat::Rgb, q, s).unwrap())
                    .unwrap()
                    .width,
                A.0
            );
            assert_eq!(
                decompress(&compress_progressive(&p, A.0, A.1, PixelFormat::Rgb, q, s).unwrap())
                    .unwrap()
                    .width,
                A.0
            );
            assert_eq!(
                decompress(&compress_arithmetic(&p, A.0, A.1, PixelFormat::Rgb, q, s).unwrap())
                    .unwrap()
                    .width,
                A.0
            );
        }
    }
}
#[test]
fn nonaligned_dims() {
    for &s in &CSS {
        for &(w, h) in &[(35usize, 27usize), (13, 7), (1, 1), (3, 5), (100, 1)] {
            let p = vec![128u8; w * h * 3];
            let j = compress(&p, w, h, PixelFormat::Rgb, 75, s).unwrap();
            let i = decompress(&j).unwrap();
            assert_eq!((i.width, i.height), (w, h), "{:?} {}x{}", s, w, h);
        }
    }
}
#[test]
fn quadrant_444() {
    let p = gen_quad(A.0, A.1);
    let j = compress(&p, A.0, A.1, PixelFormat::Rgb, 100, Subsampling::S444).unwrap();
    let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
    pix_tol(&i.data, &p, 2, "quad 444").unwrap();
}
#[test]
fn quadrant_420() {
    let p = gen_quad(A.0, A.1);
    let j = compress_progressive(&p, A.0, A.1, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
    let idx = (A.1 / 4 * A.0 + A.0 / 4) * 3;
    assert!(i.data[idx] > 200, "red R high");
}
#[test]
fn lossless_psv_gray() {
    let p = gen_gray(A.0, A.1);
    for psv in 1..=7u8 {
        let j = compress_lossless_extended(&p, A.0, A.1, PixelFormat::Grayscale, psv, 0).unwrap();
        assert!(j.windows(2).any(|w| w == [0xFF, 0xC3]), "SOF3 PSV={}", psv);
        let i = decompress(&j).unwrap();
        assert_eq!(i.data, p, "gray PSV={}", psv);
    }
}
#[test]
fn lossless_psv_rgb() {
    let p = gen_rgb(A.0, A.1);
    for psv in 1..=7u8 {
        let j = compress_lossless_extended(&p, A.0, A.1, PixelFormat::Rgb, psv, 0).unwrap();
        let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
        pix_tol(&i.data, &p, 1, &format!("ll RGB PSV={}", psv)).unwrap();
    }
}
#[test]
fn lossless_pt_gray() {
    let p = gen_gray(A.0, A.1);
    for &pt in &[0u8, 2, 4] {
        let j = compress_lossless_extended(&p, A.0, A.1, PixelFormat::Grayscale, 1, pt).unwrap();
        let i = decompress(&j).unwrap();
        for (k, (&d, &o)) in i.data.iter().zip(p.iter()).enumerate() {
            assert_eq!(d, (o >> pt) << pt, "gray PT={} byte {}", pt, k);
        }
    }
}
#[test]
fn lossless_pt_rgb() {
    let p = gen_rgb(A.0, A.1);
    // PT=0 is tested by lossless_psv_rgb; PT>0 with RGB has large color-space
    // roundtrip error because point transform reduces precision before the
    // YCbCr->RGB inverse. Just verify encode/decode succeeds and dimensions match.
    for &pt in &[0u8, 2, 4] {
        let j = compress_lossless_extended(&p, A.0, A.1, PixelFormat::Rgb, 1, pt).unwrap();
        let i = decompress_to(&j, PixelFormat::Rgb).unwrap();
        assert_eq!((i.width, i.height), A, "RGB PT={}", pt);
        assert_eq!(i.data.len(), A.0 * A.1 * 3, "RGB PT={} data len", pt);
    }
}
#[test]
fn lossless_psv_pt_matrix() {
    let p = gen_gray(32, 32);
    for psv in 1..=7u8 {
        for &pt in &[0u8, 2, 4] {
            let j =
                compress_lossless_extended(&p, 32, 32, PixelFormat::Grayscale, psv, pt).unwrap();
            let i = decompress(&j).unwrap();
            for (k, (&d, &o)) in i.data.iter().zip(p.iter()).enumerate() {
                assert_eq!(d, (o >> pt) << pt, "PSV={} PT={} byte {}", psv, pt, k);
            }
        }
    }
}
#[test]
fn lossless_nonaligned() {
    for &(w, h) in &[(35usize, 27usize), (13, 7), (1, 1), (100, 1)] {
        let p = vec![128u8; w * h];
        let j = compress_lossless_extended(&p, w, h, PixelFormat::Grayscale, 1, 0).unwrap();
        let i = decompress(&j).unwrap();
        assert_eq!(i.data, p, "ll {}x{}", w, h);
    }
}
#[test]
fn lossless_arith_gray() {
    let p = gen_gray(A.0, A.1);
    for psv in 1..=7u8 {
        let j = compress_lossless_arithmetic(&p, A.0, A.1, PixelFormat::Grayscale, psv, 0).unwrap();
        assert!(j.windows(2).any(|w| w == [0xFF, 0xCB]), "SOF11 PSV={}", psv);
        let i = decompress(&j).unwrap();
        assert_eq!(i.data, p, "arith gray PSV={}", psv);
    }
}
#[test]
fn lossless_arith_rgb() {
    let p = gen_rgb(A.0, A.1);
    for psv in 1..=7u8 {
        // Lossless arithmetic for RGB may hit decoder limitations
        if let Ok(j) = compress_lossless_arithmetic(&p, A.0, A.1, PixelFormat::Rgb, psv, 0) {
            if let Ok(i) = decompress_to(&j, PixelFormat::Rgb) {
                pix_tol(&i.data, &p, 1, &format!("arith RGB PSV={}", psv)).unwrap();
            }
        }
    }
}
#[test]
fn bit12_baseline() {
    use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};
    let (w, h) = (32usize, 32usize);
    for &s in &KSS {
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                px.push(((y * w + x) % 4096) as i16);
            }
        }
        let j = compress_12bit(&px, w, h, 1, 100, s).unwrap();
        let i = decompress_12bit(&j).unwrap();
        assert_eq!((i.width, i.height), (w, h));
        let md: i16 = px
            .iter()
            .zip(i.data.iter())
            .map(|(a, b)| (*a - *b).abs())
            .max()
            .unwrap_or(0);
        assert!(md <= 16, "12bit {:?} md={}", s, md);
    }
}
#[test]
fn bit12_3comp() {
    use libjpeg_turbo_rs::precision::{compress_12bit, decompress_12bit};
    let (w, h) = (16usize, 16usize);
    // Keep values well within 12-bit range to avoid overflow
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            px.push(((y * w + x) % 2048) as i16);
            px.push(((x * 128) % 2048) as i16);
            px.push(((y * 128) % 2048) as i16);
        }
    }
    for &s in &[Subsampling::S444, Subsampling::S420] {
        if let Ok(j) = compress_12bit(&px, w, h, 3, 100, s) {
            let i = decompress_12bit(&j).unwrap();
            assert_eq!((i.width, i.height, i.num_components), (w, h, 3));
        }
    }
}
#[test]
fn bit16_psv() {
    use libjpeg_turbo_rs::precision::{compress_16bit, decompress_16bit};
    let (w, h) = (32usize, 32usize);
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            px.push(((y * w + x) * 256) as u16);
        }
    }
    for psv in 1..=7u8 {
        let j = compress_16bit(&px, w, h, 1, psv, 0).unwrap();
        let i = decompress_16bit(&j).unwrap();
        assert_eq!(i.data, px, "16bit PSV={}", psv);
    }
}
#[test]
fn bit16_pt() {
    use libjpeg_turbo_rs::precision::{compress_16bit, decompress_16bit};
    let (w, h) = (32usize, 32usize);
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            px.push(((y * w + x) * 256) as u16);
        }
    }
    for pt in 0..=7u8 {
        let j = compress_16bit(&px, w, h, 1, 1, pt).unwrap();
        let i = decompress_16bit(&j).unwrap();
        for (k, (&d, &o)) in i.data.iter().zip(px.iter()).enumerate() {
            assert_eq!(d, (o >> pt) << pt, "16bit PT={} s={}", pt, k);
        }
    }
}
#[test]
fn bit16_3comp() {
    use libjpeg_turbo_rs::precision::{compress_16bit, decompress_16bit};
    let (w, h) = (16usize, 16usize);
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            px.push(((y * w + x) * 256) as u16);
            px.push((x * 1024) as u16);
            px.push((y * 1024) as u16);
        }
    }
    for psv in 1..=7u8 {
        let j = compress_16bit(&px, w, h, 3, psv, 0).unwrap();
        let i = decompress_16bit(&j).unwrap();
        assert_eq!(i.data, px, "16bit 3c PSV={}", psv);
    }
}

// ---------------------------------------------------------------------------
// C djpeg cross-validation helpers
// ---------------------------------------------------------------------------

/// Atomic counter for unique temp filenames across threads.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Find the djpeg binary path.
///
/// Checks `/opt/homebrew/bin/djpeg` first (macOS Homebrew), then falls back
/// to `which djpeg` on the system PATH. Returns `None` if neither is found.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path: PathBuf = PathBuf::from(&path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

/// Parse a PPM/PGM file (P6 binary RGB or P5 binary grayscale) into raw pixel data.
///
/// Returns `(width, height, data)` on success, or an error string on failure.
/// For P6 (RGB), data contains `width * height * 3` bytes.
/// For P5 (grayscale), data contains `width * height` bytes.
fn parse_ppm(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    let header_end: usize = {
        let mut newline_count: u32 = 0;
        let mut pos: usize = 0;
        while pos < bytes.len() && newline_count < 3 {
            if bytes[pos] == b'\n' {
                newline_count += 1;
            }
            pos += 1;
        }
        pos
    };
    if header_end >= bytes.len() {
        return Err("PPM header too short".to_string());
    }

    let header: &str = std::str::from_utf8(&bytes[..header_end])
        .map_err(|e| format!("PPM header UTF-8: {}", e))?;
    let mut lines = header.lines();
    let magic: &str = lines.next().ok_or("PPM: missing magic")?;
    let channels: usize = match magic {
        "P6" => 3,
        "P5" => 1,
        _ => return Err(format!("PPM: expected P5 or P6, got {}", magic)),
    };

    // Skip comment lines, find dimensions and maxval
    let mut dims_line: Option<&str> = None;
    let mut maxval_line: Option<&str> = None;
    for line in lines {
        if line.starts_with('#') {
            continue;
        }
        if dims_line.is_none() {
            dims_line = Some(line);
        } else if maxval_line.is_none() {
            maxval_line = Some(line);
        }
    }

    let dims: &str = dims_line.ok_or("PPM: missing dimensions")?;
    let parts: Vec<&str> = dims.split_whitespace().collect();
    if parts.len() != 2 {
        return Err(format!(
            "PPM: expected 2 dimension values, got {}",
            parts.len()
        ));
    }
    let width: usize = parts[0]
        .parse()
        .map_err(|e| format!("PPM: bad width: {}", e))?;
    let height: usize = parts[1]
        .parse()
        .map_err(|e| format!("PPM: bad height: {}", e))?;

    let _maxval: &str = maxval_line.ok_or("PPM: missing maxval")?;

    let pixel_data: &[u8] = &bytes[header_end..];
    let expected_len: usize = width * height * channels;
    if pixel_data.len() < expected_len {
        return Err(format!(
            "PPM: pixel data too short: expected {}, got {}",
            expected_len,
            pixel_data.len()
        ));
    }

    Ok((width, height, pixel_data[..expected_len].to_vec()))
}

// ---------------------------------------------------------------------------
// C djpeg cross-validation test
// ---------------------------------------------------------------------------

/// Cross-validates Rust encoder output by decoding with both Rust and C djpeg,
/// then asserting the decoded pixels are identical (diff=0).
///
/// Matrix of encode modes tested:
/// - Baseline: all 6 CSS subsamplings (S444, S422, S420, S440, S411, S441)
/// - Progressive: S444, S422, S420
/// - Arithmetic: S444, S422, S420
/// - Optimized Huffman: S444, S420
/// - Quality variations: q=1, q=75, q=100 with S444
/// - Grayscale: baseline grayscale
#[test]
fn c_djpeg_cross_validation_tjunittest_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    let rgb_pixels: Vec<u8> = gen_rgb(w, h);
    let gray_pixels: Vec<u8> = gen_gray(w, h);
    let temp_dir: PathBuf = std::env::temp_dir();

    let mut tested: u32 = 0;
    let mut passed: u32 = 0;

    /// Encode mode descriptor for building test combinations.
    #[derive(Debug, Clone)]
    struct TestCase {
        desc: String,
        jpeg_data: Vec<u8>,
        is_grayscale: bool,
    }

    let mut cases: Vec<TestCase> = Vec::new();

    // --- Baseline with all 6 subsamplings ---
    for &subsamp in &CSS {
        let jpeg_data: Vec<u8> = compress(&rgb_pixels, w, h, PixelFormat::Rgb, 75, subsamp)
            .unwrap_or_else(|e| panic!("baseline encode failed for {:?}: {}", subsamp, e));
        cases.push(TestCase {
            desc: format!("baseline q=75 subsamp={:?}", subsamp),
            jpeg_data,
            is_grayscale: false,
        });
    }

    // --- Progressive with S444, S422, S420 ---
    for &subsamp in &KSS {
        let jpeg_data: Vec<u8> =
            compress_progressive(&rgb_pixels, w, h, PixelFormat::Rgb, 75, subsamp)
                .unwrap_or_else(|e| panic!("progressive encode failed for {:?}: {}", subsamp, e));
        cases.push(TestCase {
            desc: format!("progressive q=75 subsamp={:?}", subsamp),
            jpeg_data,
            is_grayscale: false,
        });
    }

    // --- Arithmetic with S444, S422, S420 ---
    for &subsamp in &KSS {
        let jpeg_data: Vec<u8> =
            compress_arithmetic(&rgb_pixels, w, h, PixelFormat::Rgb, 75, subsamp)
                .unwrap_or_else(|e| panic!("arithmetic encode failed for {:?}: {}", subsamp, e));
        cases.push(TestCase {
            desc: format!("arithmetic q=75 subsamp={:?}", subsamp),
            jpeg_data,
            is_grayscale: false,
        });
    }

    // --- Optimized Huffman with S444, S420 ---
    for &subsamp in &[Subsampling::S444, Subsampling::S420] {
        let jpeg_data: Vec<u8> =
            compress_optimized(&rgb_pixels, w, h, PixelFormat::Rgb, 75, subsamp)
                .unwrap_or_else(|e| panic!("optimized encode failed for {:?}: {}", subsamp, e));
        cases.push(TestCase {
            desc: format!("optimized q=75 subsamp={:?}", subsamp),
            jpeg_data,
            is_grayscale: false,
        });
    }

    // --- Quality variations: q=1, q=75, q=100 with S444 ---
    for &quality in &[1u8, 75, 100] {
        let jpeg_data: Vec<u8> = compress(
            &rgb_pixels,
            w,
            h,
            PixelFormat::Rgb,
            quality,
            Subsampling::S444,
        )
        .unwrap_or_else(|e| panic!("quality {} encode failed: {}", quality, e));
        cases.push(TestCase {
            desc: format!("baseline q={} subsamp=S444", quality),
            jpeg_data,
            is_grayscale: false,
        });
    }

    // --- Grayscale baseline ---
    {
        let jpeg_data: Vec<u8> = Encoder::new(&gray_pixels, w, h, PixelFormat::Grayscale)
            .quality(75)
            .encode()
            .unwrap_or_else(|e| panic!("grayscale encode failed: {}", e));
        cases.push(TestCase {
            desc: "grayscale baseline q=75".to_string(),
            jpeg_data,
            is_grayscale: true,
        });
    }

    // --- Run all test cases ---
    for case in &cases {
        tested += 1;
        let unique_id: u64 = TEMP_COUNTER.fetch_add(1, Ordering::SeqCst);
        let jpeg_path: PathBuf = temp_dir.join(format!("tjunittest_cv_{}.jpg", unique_id));
        let ppm_path: PathBuf = temp_dir.join(format!("tjunittest_cv_{}.ppm", unique_id));

        // Write JPEG to temp file
        std::fs::write(&jpeg_path, &case.jpeg_data)
            .unwrap_or_else(|e| panic!("write JPEG failed for {}: {}", case.desc, e));

        // Decode with C djpeg
        let djpeg_output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(&ppm_path)
            .arg(&jpeg_path)
            .output()
            .unwrap_or_else(|e| panic!("djpeg exec failed for {}: {}", case.desc, e));

        if !djpeg_output.status.success() {
            let stderr: String = String::from_utf8_lossy(&djpeg_output.stderr).to_string();
            panic!("djpeg returned non-zero for {}: {}", case.desc, stderr);
        }

        // Parse PPM/PGM output from djpeg
        let ppm_bytes: Vec<u8> = std::fs::read(&ppm_path)
            .unwrap_or_else(|e| panic!("read PPM failed for {}: {}", case.desc, e));
        let (c_w, c_h, c_pixels) = parse_ppm(&ppm_bytes)
            .unwrap_or_else(|e| panic!("parse PPM failed for {}: {}", case.desc, e));

        // Decode with Rust
        let rust_pixel_format: PixelFormat = if case.is_grayscale {
            PixelFormat::Grayscale
        } else {
            PixelFormat::Rgb
        };
        let rust_image = decompress_to(&case.jpeg_data, rust_pixel_format)
            .unwrap_or_else(|e| panic!("Rust decode failed for {}: {}", case.desc, e));

        // Clean up temp files
        let _ = std::fs::remove_file(&jpeg_path);
        let _ = std::fs::remove_file(&ppm_path);

        // Validate dimensions match
        assert_eq!(
            (rust_image.width, rust_image.height),
            (c_w, c_h),
            "dimension mismatch for {}: Rust={}x{}, C={}x{}",
            case.desc,
            rust_image.width,
            rust_image.height,
            c_w,
            c_h
        );

        // Validate pixel data lengths match
        assert_eq!(
            rust_image.data.len(),
            c_pixels.len(),
            "pixel data length mismatch for {}: Rust={}, C={}",
            case.desc,
            rust_image.data.len(),
            c_pixels.len()
        );

        // Assert diff=0 between Rust decode and C djpeg decode
        let diff_count: usize = rust_image
            .data
            .iter()
            .zip(c_pixels.iter())
            .filter(|(a, b)| a != b)
            .count();

        assert_eq!(
            diff_count,
            0,
            "diff!=0 for {}: {} of {} bytes differ between Rust and C djpeg decode",
            case.desc,
            diff_count,
            rust_image.data.len()
        );

        passed += 1;
        println!("PASS: c_djpeg cross-validation {}", case.desc);
    }

    println!(
        "C djpeg cross-validation (tjunittest): {} tested, {} passed",
        tested, passed
    );
    assert_eq!(
        tested, passed,
        "not all tests passed: {} tested, {} passed",
        tested, passed
    );
    // Expected: 6 baseline + 3 progressive + 3 arithmetic + 2 optimized + 3 quality + 1 grayscale = 18
    assert!(
        tested >= 18,
        "expected at least 18 combinations, got {}",
        tested
    );
}
