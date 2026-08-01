# OpenCV downstream replacement experiment

**Purpose.** Deliver the first OpenCV leg of the deferred
[P2-G](../docs/last_mile/backlog.md#p2-g-downstream-lab-multi-version--multi-distro-matrix--open) downstream
compatibility lab with a real, prebuilt consumer. The test must prove that
OpenCV's JPEG calls bind to the Rust `libjpeg.so.8`; merely putting the Rust
library on `LD_LIBRARY_PATH` is not evidence that OpenCV used it.

## Environment

- Host: Linux x86_64, Docker 29.6.1.
- Base image: Ubuntu 24.04 at
  `ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90`.
- OpenCV: `libopencv-imgcodecs-dev` and its runtime libraries,
  `4.6.0+dfsg-13.1ubuntu1`.
- C baseline: Ubuntu `libjpeg-turbo8` `2.1.5-2ubuntu2`.
- Built harness image:
  `sha256:98a92909ed4d2085c036586a0d2c71928b7b728e2df3a412a0e26784af89e227`
  (760,392,010 bytes).
- Rust candidate: the release cdylib from the current source tree, staged as
  `/tmp/libjpeg-rs/libjpeg.so.8`.

The base digest, OpenCV development-package version, expected OpenCV ABI
suffix (`.so.406`), and expected OpenCV version are pinned by the harness.
Other transitive Ubuntu packages still come from the live Noble repositories,
so the image ID above records the exact measured environment.

## Workload and falsification checks

`examples/opencv_smoke/main.cpp` creates a deterministic 257x193 BGR image.
The odd dimensions exercise right and bottom MCU tails. It then uses real
OpenCV APIs to:

1. write quality-90 progressive JPEG with Huffman optimization and restart
   interval 4 via `cv::imwrite`;
2. read it through both `IMREAD_COLOR` and `IMREAD_GRAYSCALE`;
3. require the original dimensions and OpenCV types, PSNR >= 49.22 dB (a
   0.006 dB margin below the pinned system baseline), and the measured
   grayscale checksum 6,299,471; and
4. cross-decode the other implementation's JPEG in both color modes.

The container runner additionally requires all of the following:

- the mounted candidate exports `tj3InitVersion`, which the Rust C ABI shim
  carries but Ubuntu's system `libjpeg.so.8` does not;
- the candidate's own dynamic section advertises `DT_SONAME=libjpeg.so.8`,
  rather than relying on the staging symlink to relabel a v6b build;
- the packaged `libopencv_imgcodecs.so.406` has a dynamic dependency on
  `libjpeg.so.8`;
- `ldd` resolves the smoke executable's JPEG dependency to the staged Rust
  library under `LD_LIBRARY_PATH`; and
- glibc `LD_DEBUG=bindings` records `libopencv_imgcodecs.so.406` binding both
  `jpeg_CreateCompress` and `jpeg_CreateDecompress` to the staged Rust
  `libjpeg.so.8`; and
- the persisted BGR and grayscale matrices compare byte-for-byte between the
  system and Rust decoders for both the system-produced and Rust-produced
  JPEGs.

This prevents a false green where OpenCV silently uses its system JPEG library
or a built-in codec. As a negative control, mounting Ubuntu's own
`/lib/x86_64-linux-gnu/libjpeg.so.8` as the candidate exits 13 before the
workload because it lacks the candidate's `tj3InitVersion` surface.

## Command

```bash
cargo build -p libjpeg-turbo-rs-capi --release
bash examples/opencv_smoke/run.sh \
  --lib target/release/liblibjpeg_turbo_rs_capi.so \
  --workdir target/opencv-smoke
```

## Result

| path | PSNR | grayscale sum |
| --- | ---: | ---: |
| system encode + system decode | 49.226 dB | 6,299,471 |
| Rust encode + Rust decode | 49.226 dB | 6,299,471 |
| system encode + Rust decode | 49.226 dB | 6,299,471 |
| Rust encode + system decode | 49.226 dB | 6,299,471 |

Both outputs are progressive, 8-bit, three-component JFIF images at 257x193.
After the C ABI restart-setting fix, the system and Rust encoders produce
byte-identical JPEGs. The harness enforces that identity in addition to the
bidirectional decoded-matrix comparisons.

```text
system.jpg  2945f085182223131779686ca88c83d0ee816222a1517bd289946ac106316905
rust.jpg    2945f085182223131779686ca88c83d0ee816222a1517bd289946ac106316905
```

The loader log contains the decisive bindings:

```text
binding file /lib/x86_64-linux-gnu/libopencv_imgcodecs.so.406 ...
  to /tmp/libjpeg-rs/libjpeg.so.8 ... `jpeg_CreateCompress' [LIBJPEG_8.0]
binding file /lib/x86_64-linux-gnu/libopencv_imgcodecs.so.406 ...
  to /tmp/libjpeg-rs/libjpeg.so.8 ... `jpeg_CreateDecompress' [LIBJPEG_8.0]
```

**Verdict: pass.** A stock Ubuntu OpenCV 4.6 binary successfully performs
JPEG encode, color decode, grayscale decode, and both cross-implementation
decode directions through the Rust C ABI shim.

## Failed iterations retained as evidence

1. The first compile passed the entire `pkg-config --cflags-only-I` output as
   one quoted argument, so `g++` could not find `opencv2/core.hpp`. The runner
   now uses the pinned Ubuntu include directory directly.
2. The first fixture changed channels by 17-19 levels per pixel and included a
   small checkerboard. Even the system baseline measured only 17.9204 dB at
   quality 90. That tested an adversarial signal rather than downstream
   replacement, so the fixture was changed to gradients plus 32x24 tiles. The
   system baseline now measures 49.226 dB; review then replaced the original
   loose 28 dB floor with 49.22 dB and pinned the measured grayscale checksum.
3. The first green OpenCV run requested restart interval 4 but did not inspect
   the JPEG structure. Review found that `system.jpg` carried DRI=4 and RST
   markers while `rust.jpg` carried neither: the classic C scanline shim
   discarded both public restart fields. Structural assertions made the
   harness red; forwarding block-mode and row-mode restart settings through
   every pixel-encode entropy branch made it green and byte-identical.

## Limits and follow-ups

- This is one OpenCV version, one Ubuntu release, glibc, x86_64, and one
  `imgcodecs` workload. Qt5/Qt6 and the P2-G multi-version/multi-distro matrix
  remain open.
- The version-pinned manual harness is rerunnable, but the locally recorded
  760 MB image ID is not a permanent artifact and transitive packages still
  come from live Ubuntu repositories. A scheduled matrix should publish a
  registry image or use a package cache before expanding coverage.
- Ubuntu's OpenCV and several transitive libraries request versioned
  `LIBJPEG_8.0` symbols. The Rust cdylib currently exports unversioned symbols,
  so glibc resolves them but prints `no version information available`.
  Functional binding is proven here; warning-free ELF replacement is tracked
  separately as P4-81.
