# Long-Term Replacement Backlog

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). This file keeps deferred
> replacement work in the repository so its scope and acceptance criteria do
> not depend on a developer-machine plan.

## P2-G. Downstream Lab Multi-Version / Multi-Distro Matrix — **OPEN**

**Purpose.** Turn the existing one-version downstream smoke tests into a
scheduled compatibility lab that catches ABI, loader, packaging, and behavior
regressions across real prebuilt consumers. This is a coverage program, not a
claim that every listed consumer blocks each release.

**Progress (2026-08-02).** The first OpenCV leg is complete in
`examples/opencv_smoke/`: pinned Ubuntu 24.04 and OpenCV 4.6, real
`cv::imwrite`/`cv::imread`, loader-binding proof, structural JPEG assertions,
and byte-exact system/Rust self- and cross-decode comparisons. Its measured
record is `experiments/opencv_downstream_2026-08-02.md`. The run found and
closed P4-82 (classic scanline restart settings were dropped) and filed P4-81
(GNU ELF symbol versions are missing). The ensuing dispatcher review found and
closed baseline smoothing P4-83 and filed the still-open
progressive/arithmetic smoothing composition as P4-84.

**Remaining matrix.** Add Qt5 and Qt6 image-plugin workloads; exercise at
least two supported OpenCV versions and two Linux distro generations; and
extend the existing Pillow 10/11, ImageMagick 6/7, libvips 8.x, FFmpeg 6/7,
and libtiff 4.x legs where version coverage is still single-point.

**Acceptance criteria.** (1) Every leg proves that the intended Rust cdylib,
not a system fallback or built-in codec, handled the JPEG calls; (2) encode,
decode, and cross-implementation paths compare pixels or bytes under an
explicit oracle; (3) missing packages and empty test selection fail rather
than skip silently; (4) scheduled runs use immutable registry images or a
package cache with recorded package inventories; and (5) failures preserve
the loader evidence, inputs, outputs, and exact environment needed to
reproduce them.
