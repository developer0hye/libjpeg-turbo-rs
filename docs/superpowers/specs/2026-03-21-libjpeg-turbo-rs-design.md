# libjpeg-turbo-rs Design Spec

## Goal

Pure Rust reimplementation of libjpeg-turbo with equivalent or better performance. Full JPEG spec coverage including encoding and decoding.

## Key Decisions

- **Pure Rust** — no C/assembly dependencies, SIMD via `std::arch` intrinsics
- **Decoding first** — build common modules (DCT, color conversion, Huffman, SIMD backend), then reuse for encoding
- **SIMD targets** — AArch64 NEON first (dev machine: Apple M2), x86_64 SSE2/AVX2 via CI, scalar fallback always available
- **Full JPEG spec** — Baseline, Progressive, Arithmetic coding, Lossless, 8/12/16-bit
- **Dual API** — high-level TurboJPEG-style + streaming libjpeg-style, internal pipeline is streaming
- **Unsafe policy** — safe Rust by default; unsafe only for SIMD intrinsics and proven hot-path bounds check elision; every `unsafe` block requires `// SAFETY:` justification

## Architecture

```
┌─────────────────────────────────┐
│  High-level API (TurboJPEG式)   │  compress() / decompress() / transform()
├─────────────────────────────────┤
│  Streaming API (libjpeg式)      │  scanline-by-scanline read/write
├─────────────────────────────────┤
│  Core Pipeline                  │  each module abstracted via traits
│  ┌───────┬──────┬───────┬─────┐ │
│  │Color  │DCT/  │Huffman│Sample│ │
│  │Convert│IDCT  │Coding │Up/Dn │ │
│  └───┬───┴──┬───┴───┬───┴──┬──┘ │
│      │      │       │      │    │
│  ┌───▼──────▼───────▼──────▼──┐ │
│  │  SIMD Backend (per-arch)   │ │
│  │  AArch64 NEON | x86 SSE2/ │ │
│  │  AVX2 | Scalar fallback    │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
```

## Crate Structure

```
libjpeg-turbo-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # public API re-exports
│   ├── api/
│   │   ├── high_level.rs      # compress(), decompress(), transform()
│   │   └── streaming.rs       # scanline-by-scanline Decoder/Encoder
│   ├── decode/
│   │   ├── marker.rs          # marker parsing (SOF, SOS, DHT, DQT, APP, COM)
│   │   ├── huffman.rs         # Huffman decoding
│   │   ├── arithmetic.rs      # arithmetic decoding
│   │   ├── dequant.rs         # dequantization
│   │   ├── idct.rs            # inverse DCT
│   │   ├── upsample.rs        # chroma upsampling
│   │   ├── color.rs           # YCbCr → RGB color conversion
│   │   ├── progressive.rs     # progressive JPEG
│   │   └── lossless.rs        # lossless JPEG
│   ├── encode/
│   │   ├── marker.rs          # marker writing
│   │   ├── huffman.rs         # Huffman encoding
│   │   ├── arithmetic.rs      # arithmetic encoding
│   │   ├── quant.rs           # quantization
│   │   ├── fdct.rs            # forward DCT
│   │   ├── downsample.rs      # chroma downsampling
│   │   ├── color.rs           # RGB → YCbCr color conversion
│   │   ├── progressive.rs     # progressive encoding
│   │   └── lossless.rs        # lossless encoding
│   ├── common/
│   │   ├── types.rs           # ColorSpace, Subsampling, PixelFormat, etc.
│   │   ├── error.rs           # JpegError enum
│   │   ├── huffman_table.rs   # shared Huffman tables
│   │   └── quant_table.rs     # shared quantization tables
│   └── simd/
│       ├── mod.rs             # runtime dispatch via SimdBackend trait
│       ├── scalar.rs          # fallback implementation
│       ├── aarch64.rs         # NEON intrinsics
│       └── x86_64.rs          # SSE2/AVX2 intrinsics
├── tests/
│   └── conformance/           # bit-exact verification against libjpeg-turbo
├── benches/
│   └── decode.rs              # criterion benchmarks
└── fuzz/
    └── decode.rs              # cargo-fuzz harness
```

## SIMD Strategy

```rust
pub trait SimdBackend {
    fn idct_8x8(&self, coeffs: &[i16; 64], output: &mut [u8; 64]);
    fn ycbcr_to_rgb(&self, y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8]);
    fn upsample_h2v2(&self, input: &[u8], output: &mut [u8]);
    // per hot-path method
}

// compile-time arch selection + runtime feature detection (x86_64 AVX2)
pub fn detect() -> &'static dyn SimdBackend { ... }
```

- `#[cfg(target_arch = "aarch64")]` → NEON (always available on AArch64)
- `#[cfg(target_arch = "x86_64")]` → SSE2 baseline + AVX2 runtime detection
- Scalar fallback always compiled

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum JpegError {
    #[error("invalid marker: 0x{0:02X}")]
    InvalidMarker(u8),
    #[error("unsupported feature: {0}")]
    Unsupported(String),
    #[error("corrupt data: {0}")]
    CorruptData(String),
    #[error("buffer too small: need {need}, got {got}")]
    BufferTooSmall { need: usize, got: usize },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

## Public API

### High-level

```rust
let pixels: RgbImage = libjpeg_turbo_rs::decompress(&jpeg_bytes)?;
let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(&pixels, Quality(85))?;
```

### Streaming

```rust
let mut decoder = Decoder::new(reader)?;
let header = decoder.header(); // width, height, colorspace, subsampling
while let Some(scanline) = decoder.next_scanline()? {
    // process scanline
}
```

## Testing Strategy

- **Conformance** — decode libjpeg-turbo test images, bit-exact comparison
- **Fuzz** — `cargo-fuzz` with malformed JPEG inputs
- **Benchmark** — `criterion` comparing against libjpeg-turbo (C) and zune-jpeg
- **SIMD verification** — each SIMD path must produce identical output to scalar fallback

## Implementation Roadmap

| Phase | Scope | Goal |
|-------|-------|------|
| 1 | Baseline JPEG decoder (scalar) | Correctness |
| 2 | SIMD optimization (NEON first, then SSE2/AVX2) | libjpeg-turbo-level performance |
| 3 | Progressive + Arithmetic decoding | Full decoding spec |
| 4 | Lossless JPEG + 12/16-bit decoding | Complete decoding |
| 5 | Baseline encoder (scalar → SIMD) | Encoding begins |
| 6 | Progressive + Arithmetic + Lossless encoding | Full encoding spec |
| 7 | Lossless transform | Full spec complete |
