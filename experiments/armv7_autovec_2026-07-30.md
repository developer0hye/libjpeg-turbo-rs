# armv7: the target baseline disables auto-vectorisation (2026-07-30)

Follow-up to `x86_64_scalar_2026-07-30.md`, which found our "scalar" IDCT
contains 162 packed SIMD instructions on x86_64 because SSE2 is in that
ABI's baseline. The obvious next question: **is the same vectoriser simply
switched off on ARM?** It is.

**This is a finding about a missing build flag, not an optimisation we
wrote.** And it is explicitly *not* a recommendation — see "Why this must
stay opt-in".

## Environment

| | |
| --- | --- |
| Host | Windows 11 + WSL2, Ubuntu 24.04.4, i5-1334U |
| Emulator | `qemu-arm-static` 8.2.2 (`-L /usr/arm-linux-gnueabihf`) |
| Toolchain | rustc 1.97.1, `armv7-unknown-linux-gnueabihf`, `arm-linux-gnueabihf-gcc` 13.3.0 |
| Tree | 0.8.0 @ `0df6efb` |
| Baseline build | `cargo build --release --target armv7-unknown-linux-gnueabihf` |
| Test build | same, `RUSTFLAGS="-C target-feature=+neon"` |

## Instruction-level evidence

`arm-linux-gnueabihf-objdump -d --demangle`, counting vector mnemonics
(`vadd|vsub|vmul|vshr|vshl|vld1|vst1|vqmov|vmovl|vmlal|vzip|vuzp`) inside
each kernel's body:

| kernel | armv7 default | armv7 `+neon` |
| --- | --- | --- |
| `decode::idct::idct_8x8` | 0 | **270** |
| `decode::color::ycbcr_to_rgb_row` | 0 | **140** |
| `decode::upsample::fancy_h2v2_row` | 0 | **232** |
| whole binary, 128-bit `q` register operands | **0** | **2077** |

The default build's 222 hits on a looser `v*` pattern are VFP *scalar*
double ops on `d` registers, not NEON — the `q`-register count (0) is the
unambiguous figure.

Note `ycbcr_to_rgb_row` vectorises on ARM but **not** on x86_64 (0 packed
instructions there): the table-driven form from P4-60 step 1 maps onto
NEON's byte-lane handling in a way LLVM would not do with SSE2.

## Correctness

204/204 green under `qemu-arm-static` with `+neon` **and**
`-C overflow-checks=on` (196 lib + 6 `simd_dispatch` + 2
`no_std_dispatch`), via
`CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUNNER`.

## A/B — and why the number is nearly meaningless

`examples/bench_scalar_p460` in-process medians, `taskset -c 0`, two reps:

| case | default | `+neon` | ratio |
| --- | --- | --- | --- |
| `photo_640x480_420` | 16,401 / 16,966 us | 14,041 / 14,366 us | **1.17x** |
| `photo_1920x1080_420` | 135,067 / 136,292 us | 115,852 / 119,374 us | **1.17x** |

The comparison form is legitimate — ours-before vs ours-after, same
emulator, so the emulation tax is symmetric, unlike an ours-vs-C-with-NEON
comparison. What it cannot be is a hardware prediction: **QEMU models no
pipeline, no cache, and no register-domain transfer cost.**

## Why this must stay opt-in

1. **`target-feature` is compile-time.** A `+neon` binary **SIGILLs** on an
   ARMv7 core without NEON. C ships one binary for both by probing
   `/proc/cpuinfo` (`simd/arm/aarch32/jsimdcpu.c:72-125`); a compile-time
   flag structurally cannot.
2. **It may be slower on real silicon**, in ways this environment is blind
   to by construction:
   - **Cortex-A8**: NEON sits behind the integer pipeline with an expensive
     NEON->ARM register transfer. Vectoriser prologues, reductions and tail
     loops generate exactly that boundary traffic.
   - **Cortex-A7 / A9**: 64-bit NEON datapath, so 128-bit `q` ops issue over
     two cycles — setup and tail cost is proportionally larger than on a
     128-bit A15.
   - **No `-C target-cpu=`**: the vectoriser is costing against a *generic*
     ARMv7 model, so its decisions are not tuned for whatever core ships.
   - **Short trip counts**: an 8-point IDCT row is near the width where
     vectorisation setup can exceed the gain.

C's hand-written AArch32 NEON kernels being a proven win says nothing about
auto-vectorised output, which is not written to dodge those traps.
**Read 1.17x as "vectorisation happens", not "vectorisation pays".**

## To turn this into a real recommendation

- Hardware A/B on the actual target core, with `-C target-cpu=` naming that
  core instead of generic ARMv7.
- **Per-kernel** A/B, not whole-decode: an average can hide one kernel
  regressing while others improve.
- Compare against `+neon` *plus* `-C target-cpu=` separately — the cost
  model change alone may matter more than the feature bit.
- Keep the flag documented next to the existing x86_64 build-flag guidance,
  never as a default, and always with the SIGILL caveat attached.

Tracked as option (D) in P4-78.
