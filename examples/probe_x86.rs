fn main() {
    let bytes = std::fs::read("/tmp/fuzz_smoke_crash.jpg").expect("read");
    let raw = libjpeg_turbo_rs::decompress_raw(&bytes).expect("decompress_raw");
    println!("comps={}", raw.num_components);
    for (i, name) in ["Y", "Cb", "Cr"].iter().enumerate().take(raw.num_components) {
        let p = &raw.planes[i];
        let mn = *p.iter().min().unwrap();
        let mx = *p.iter().max().unwrap();
        println!("{}: min={} max={} row0={:?}", name, mn, mx, &p[..16]);
    }
}
