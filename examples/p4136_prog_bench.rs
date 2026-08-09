//! P4-136 criterion 3: zero-init vs uninit on the ProgressiveDecoder path.
use std::time::Instant;
fn main() {
    let path = "tests/fixtures/real_world/derived_7680x4320_8k_progressive.jpg";
    let data = std::fs::read(path).expect("fixture");
    // warm-up
    for _ in 0..2 {
        let mut d = libjpeg_turbo_rs::ProgressiveDecoder::new(&data).unwrap();
        while d.consume_input().unwrap() {}
        let _ = d.output().unwrap();
    }
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        let mut d = libjpeg_turbo_rs::ProgressiveDecoder::new(&data).unwrap();
        while d.consume_input().unwrap() {}
        let img = d.output().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&img);
        if ms < best {
            best = ms;
        }
    }
    println!("PROG_BEST_MS {best:.2}");
}
