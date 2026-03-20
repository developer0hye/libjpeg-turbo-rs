use criterion::{criterion_group, criterion_main, Criterion};

fn bench_decode(_c: &mut Criterion) {
    // TODO: add benchmarks after conformance tests pass
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
