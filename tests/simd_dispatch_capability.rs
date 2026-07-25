//! Issue #320: makes the SIMD fallback CI leg self-verifying.
//!
//! The `SSE2-only` job set `RUSTFLAGS: -C target-feature=-avx2` and claimed to
//! validate the AVX2-unavailable path. It did not: that flag is compile-time,
//! while every SIMD dispatch here is a runtime `is_x86_feature_detected!`
//! CPUID query, which ignores it. On a runner with AVX2 the job therefore
//! exercised the AVX2 path — with a name and comment asserting the opposite.
//!
//! A job that emulates an older CPU has the same failure mode available to it:
//! if the emulation silently does not mask the feature, the leg passes while
//! testing nothing. So the expectation is asserted rather than assumed.
//!
//! Set `EXPECT_NO_AVX2=1` when running under an emulated pre-AVX2 CPU
//! (`qemu-x86_64 -cpu Nehalem`, Intel SDE `-snb`, ...). The test then fails if
//! AVX2 is still visible, which means the leg is not testing what it says.

/// Reports the runtime dispatch state, and enforces `EXPECT_NO_AVX2` when set.
///
/// Deliberately not `#[cfg(target_arch = "x86_64")]`-only at the file level:
/// on other architectures it degrades to a no-op with an explicit message
/// rather than silently vanishing from the run.
#[test]
fn simd_dispatch_matches_the_expected_capability() {
    let expect_no_avx2: bool = std::env::var("EXPECT_NO_AVX2")
        .map(|value| value == "1")
        .unwrap_or(false);

    #[cfg(target_arch = "x86_64")]
    {
        let avx2: bool = is_x86_feature_detected!("avx2");
        let sse2: bool = is_x86_feature_detected!("sse2");
        let ssse3: bool = is_x86_feature_detected!("ssse3");
        eprintln!(
            "runtime dispatch: avx2={avx2} sse2={sse2} ssse3={ssse3} \
             (compile-time avx2={})",
            cfg!(target_feature = "avx2")
        );

        if expect_no_avx2 {
            assert!(
                !avx2,
                "EXPECT_NO_AVX2=1 but is_x86_feature_detected!(\"avx2\") is still \
                 true — the emulated CPU is not masking AVX2, so this leg is \
                 exercising the AVX2 path and testing nothing it claims to \
                 (issue #320)"
            );
            // SSE2 is baseline on x86_64; if it were missing the fallback under
            // test would not be the SSE2 fallback either.
            assert!(
                sse2,
                "EXPECT_NO_AVX2=1 and SSE2 is also unavailable — this is below \
                 the x86_64 baseline and not the configuration under test"
            );
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        assert!(
            !expect_no_avx2,
            "EXPECT_NO_AVX2=1 was set on a non-x86_64 target ({}), where it is \
             meaningless — the CI leg is misconfigured",
            std::env::consts::ARCH
        );
        eprintln!(
            "SKIP: x86_64 capability dispatch not applicable on {}",
            std::env::consts::ARCH
        );
    }
}

/// Encodes through the paths that have historically differed between the AVX2
/// and fallback kernels, so the emulated leg exercises them rather than merely
/// reporting the CPU features.
///
/// #314 (dummy-block contract) and #330 (DCT-method gating) both lived in
/// AVX2-only code, and #330's fallback branch is what runs here.
#[test]
fn simd_sensitive_encode_paths_round_trip() {
    use libjpeg_turbo_rs::{decompress_to, DctMethod, Encoder, PixelFormat, Subsampling};

    let (width, height) = (33usize, 18usize); // partial MCU in both axes
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x3f) - 32;
            let offset: usize = (y * width + x) * 3;
            pixels[offset] = ((x * 255 / width) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 1] = ((y * 255 / height) as i32 - noise).clamp(0, 255) as u8;
            pixels[offset + 2] = (((x ^ y) & 0xff) as i32 + noise).clamp(0, 255) as u8;
        }
    }

    for subsampling in [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
    ] {
        for dct_method in [DctMethod::IsLow, DctMethod::IsFast, DctMethod::Float] {
            let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .subsampling(subsampling)
                .dct_method(dct_method)
                .encode()
                .unwrap_or_else(|error| {
                    panic!("{subsampling:?} {dct_method:?} encode failed: {error:?}")
                });

            let decoded = decompress_to(&jpeg, PixelFormat::Rgb)
                .unwrap_or_else(|error| panic!("{subsampling:?} decode failed: {error:?}"));
            assert_eq!(decoded.width, width);
            assert_eq!(decoded.height, height);

            // A mis-scaled divisor table (the #330 shape) shows up as gross
            // error, well beyond ordinary quantization loss at q75.
            let mean_error: f64 = decoded
                .data
                .iter()
                .zip(pixels.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
                .sum::<u64>() as f64
                / pixels.len() as f64;
            assert!(
                mean_error < 20.0,
                "{subsampling:?} {dct_method:?}: mean error {mean_error:.2} is far \
                 beyond q75 quantization loss — a fallback kernel is mis-scaled"
            );
        }
    }
}
