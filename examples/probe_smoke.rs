fn main() {
    let bytes = std::fs::read("/tmp/fuzz_smoke_crash.jpg").expect("read");
    let raw = libjpeg_turbo_rs::decompress_raw(&bytes).expect("decompress_raw");
    for (i, name) in ["Y", "Cb", "Cr"]
        .iter()
        .enumerate()
        .take(raw.num_components)
    {
        let p = &raw.planes[i];
        println!(
            "{}: min={} max={}",
            name,
            *p.iter().min().unwrap(),
            *p.iter().max().unwrap()
        );
    }
}
