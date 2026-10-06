# Known memory-safety and abort findings by published version

Prepared 2026-10-07 for #635, Milestone A: "Review affected published versions and disclosure needs".
Repository state: `origin/main` @ `7f9e5e5`. This is a read-only analysis. No finding was reproduced or executed for this table. Each boundary comes from git history, `git tag --contains`, and a grep of the **published crate tarballs** downloaded from crates.io (see section 1).

## 1. What was actually published

Every `.crate` file was downloaded and its `src/` was diffed against the repository tree. The `.cargo_vcs_info.json` SHAs in the tarballs (for example `ee3f3de` for 0.8.0) are not in the current local history. GitHub still serves them as unreachable commits. *Inference, not verified:* the history was rewritten after publishing, because the SHAs differ while the trees are identical. Commit SHAs alone therefore cannot locate a published version. Tree contents can, and the source trees match as follows:

| Package (crates.io) | Published | Source tree that equals the published `src/` |
|---|---|---|
| `libjpeg-turbo-rs` 0.1.0, 0.1.1, 0.2.0, 0.3.0, 0.4.0 | 03-29 … 04-09 | tags `v0.1.0` … `v0.4.0` (0 diffs each) |
| `libjpeg-turbo-rs` **0.2.1** | 2026-03-30 | **untagged** `7006fd5` (merge of #118, between `v0.2.0` and `v0.3.0`) |
| `libjpeg-turbo-rs` **0.5.0** | 2026-04-11 | **not `v0.5.0`** (37 files differ). Equals `954127e` "bump version to 0.5.0", the day before the tag. |
| `libjpeg-turbo-rs` 0.6.0 … 0.8.0 | 05-03 … 07-28 | tags `v0.6.0` … `v0.8.0` (0 diffs each) |
| `libjpeg-turbo-rs-capi` **0.1.0** | 2026-05-06 | **`v0.6.1`** capi tree (0 diffs). It is *not* `v0.6.0` (3 diffs), because the publish job (`dc0b298`) landed after `v0.6.0`. Depends on core `^0.6`. |
| `libjpeg-turbo-rs-capi` 0.1.1 | 2026-07-26 | `v0.7.0` (0 diffs). Depends on core `^0.7`. |
| `libjpeg-turbo-rs-capi` 0.1.2 | 2026-07-28 | `v0.8.0` (0 diffs). Depends on core `^0.8`. |
| `libjpeg-turbo-rs-image` 0.1.0 | 2026-07-28 | `v0.8.0` (0 diffs). Depends on core `0.8.0`. Calls only `decompress_to` / `compress`. |

Consequences:
- The `v0.6.2` and `v0.6.3` tags carry capi 0.1.0's version number, but no capi was published from them.
- A fix whose first tag is `v0.6.2` (P4-4 and P4-3) first reached crates.io in **capi 0.1.1**.
- Published core 0.5.0 does **not** contain P4-41.

Every August-2026 soundness fix is unreleased. No published root-crate version carries any of the #481 fixes, and no published capi version carries any of the August C-ABI fixes. Every August-2026 fix in the table merged after `v0.8.0` (2026-07-29) and is in no tag. The only fixes that did reach a tag are P4-41 (in `v0.7.0`) and P4-3/P4-4 (in `v0.6.2`, published as capi 0.1.1).

Version shorthand in the table:
- "core all" = 0.1.0, 0.1.1, 0.2.0, 0.2.1, 0.3.0, 0.4.0, 0.5.0, 0.6.0, 0.6.1, 0.6.2, 0.6.3, 0.7.0, 0.8.0.
- "capi all" = 0.1.0, 0.1.1, 0.1.2.

## 2. Disclosure table

"Advisory warranted?" means:
- **yes**: memory unsafety reachable from safe Rust, or from documented-correct C-ABI use, in a published version.
- **consider**: process abort or DoS, or memory unsafety that needs an unusual target or configuration.
- **no**: anything else.

### 2a. Safe Rust API (`libjpeg-turbo-rs`)

| Finding | Reachable from | Class | Introduced (commit, first tag) | Fixed (PR/commit, first tag) | Affected published versions | Advisory warranted? |
|---|---|---|---|---|---|---|
| **P4-41 / #315**: the AVX2 4:2:0 row fast path in `compress()` / `Encoder` gated on `!cb_half.is_empty()` instead of an AVX2 check, so non-AVX2 x86_64 CPUs entered `#[target_feature(enable="avx2")]` helpers | Safe Rust: any 4:2:0 encode on an x86_64 CPU without AVX2. Also C ABI through capi 0.1.0, whose `^0.6` requirement resolves to an affected core. That chain is `tj3Compress8` → `TjHandle::compress` → `Encoder::encode`; whether `Encoder` defaults select the fast path was not traced. | UB (executes AVX2 code without AVX2; in practice SIGILL) | `c03bdbb` 2026-04-12, first tag `v0.5.0`. Published 0.5.0 (`954127e`) predates it, so it was first published in **0.6.0**. | PR #318 (`778aecc`, merge `9f54857`, 2026-07-25), **`v0.7.0`** | core **0.6.0, 0.6.1, 0.6.2, 0.6.3**; capi 0.1.0 (via core 0.6.x) | **yes**. This is the only finding that is fixed *and* has a published fixed version (0.7.0). |
| **P4-135 / #474**: public safe `pub fn` wrappers (`simd::x86_64::avx2_color::*` and others, plus the `pub` fn-pointer fields of `simd::detect()`) call `target_feature` kernels with an unchecked `width` and no CPU-feature check | Safe Rust only. No `unsafe` is needed at the call site, as shown by a compile-checked probe. | UB: out-of-bounds read/write, and AVX2 code without AVX2 | `d67314e` 2026-03-22, `v0.1.0`. `pub mod simd` and the wrapper comment "dispatch verifies this" appear in every published core. | #503 + #507 (2026-08-09/10), #554 + #555 (2026-08-13). Unreleased (main). | core all (x86_64 confirmed). Other architectures: see note 3. | **yes** |
| **P4-136 / #475**: `ProgressiveDecoder::output()` calls `Vec::set_len` on uninitialized `Vec<u8>` planes, and the plane sizes come from unchecked multiplication | Safe Rust (`ProgressiveDecoder`, public since `v0.1.0`). Not reached by `decompress()`, the image adapter or the C ABI. Verified by grep: no `ProgressiveDecoder`/`progressive_output` reference in the capi `src/` at `v0.6.1` or `v0.8.0`; classic buffered-image mode does not route through it. | UB: violates the `set_len` contract on every target. Phase4 also claims a wrapped-small allocation the IDCT then writes past, but only 32-bit targets can wrap; see note 4. | `f23470b` 2026-03-22, `v0.1.0` | #487 (`185c990`, 2026-08-09) and #511 (`7a55010`, 2026-08-10). Unreleased (main). | core all (6–7 `set_len` sites in every published version) | **yes** ("unsound"). Practical exploitability is unverified. |
| **P4-192 / #610**: `Encoder::scan_script` accepts a custom script without validation. `Se > 63` (or `Se < Ss`, which wraps `band_len`) drives `prepare_ac_first_sse2` past two `[u16; 64]` stack arrays, and the call returns `Ok`. | Safe Rust only, on x86_64 (SSE2 is baseline, so no feature gate applies). Triggered by a caller-supplied option, not by file bytes. Not reachable through the C ABI (P4-91: classic `scan_info` is ignored). | UB: stack buffer overflow (write) | `1f5e683` 2026-04-08 (SSE2 prepare on the custom path), `v0.4.0`. The custom-script API itself dates from `v0.1.0`, but 0.1.0–0.3.0 have no unchecked kernel on that path (note 5). | PR #639 (fix and P4-211). Unreleased (main once merged). | core **0.4.0, 0.5.0, 0.6.0–0.6.3, 0.7.0, 0.8.0** (x86_64) | **yes** |
| **P4-138 / #477**: `BitWriter` hand-rolled raw ownership. An unwinding `reserve` in `ensure_capacity` double-frees, and `BitWriter` was `pub`. | Safe Rust (every encode, and `encode::huffman_encode::BitWriter` directly) | UB (double free), but only if growth *unwinds* | `c4287b0` 2026-03-27, `v0.1.0` | #489 (`0504c5b`, 2026-08-09) and #513 (`09908e2`, 2026-08-10). Unreleased (main). | core all, in theory (note 6) | **consider**. Not reachable on 64-bit with default alloc/panic behaviour. On 32-bit with `panic=unwind` it needs more than 1 GiB of entropy output. Confirmed under fault injection only. |
| **P4-209 / #632**: the mainline decode destination (`decompress`, `Decoder::decode_image`) uses an infallible `vec![0u8; size]`, so an allocator refusal aborts the process | Safe Rust, the image adapter, the C ABI (`tj3Decompress8` etc.) and the wasm npm package | Process abort (DoS). The amplifying size is SOF-derived, up to the default `max_pixels` of 2³¹−1 (about 6–8.6 GB of RGB/RGBA output). | Initial decoder, `v0.1.0`. No published core contains a single `try_reserve`. | Fix on branch `fix/p4-209-fallible-decode-alloc` (PR pending); 12/16-bit paths remain, P4-216. Unreleased. | core all; image 0.1.0; capi all | **consider** |
| **P4-136 criterion 4 / P4-144 / P4-153**: progressive geometry-sized allocations (amplifying) and metadata copies (ICC/EXIF/XMP/markers; not amplifying) are infallible | Safe Rust and C ABI | Process abort (DoS) | `v0.1.0` (same shape as P4-209) | #511 (geometry, 2026-08-10), #537 (`005869c`, 2026-08-12), #542 (2026-08-12). Unreleased (main). | core all; capi all; image 0.1.0 (metadata path) | **consider**. Low for the metadata half, which is bounded by input size. |
| **P4-199 / #620**: `TjHandle::decompress_12bit/16bit` (and `tj3Decompress12/16`) ignore the handle's `MAXPIXELS`/`MAXMEMORY`/`SCANLIMIT` | Safe Rust (`TjHandle`) and C ABI | Resource-limit bypass, which leads to large allocation and abort (DoS) | `7cbe769` 2026-04-16 (`TjHandle::decompress_16bit`) and `b81a683` 2026-04-18 (`tj3Decompress16`), `v0.6.0` | **Open** | core 0.6.0–0.8.0; capi all | **consider** |
| **P4-197 / #618**: a crop region with `x` ≥ scaled width decodes to zero columns. Debug builds panic on a `debug_assert!`; release returns `Ok` and drops the crop height. | Safe Rust (`set_crop_region`, `TjHandle`) and C ABI (`tj3SetCroppingRegion`) | Other: contract/logic error. Debug-only panic, no memory unsafety. | Clamp code `4aa40fd` 2026-04-14, `v0.6.0` (not present in 0.1.0–0.5.0) | **Open** | core 0.6.0–0.8.0; capi all | no |
| **P4-139 / #478** (PARTIAL): saturating/unchecked span arithmetic. `ScalingFactor` has public fields that panic. | C ABI (saturated `pitch*h` slice length) and safe Rust (`ScalingFactor` panics) | Other/hardening. A saturated span needs an impossible `pitch*height` (caller-invalid arguments). `ScalingFactor` panics on caller-chosen values. | capi spans `85c6b2a` 2026-04-18, `v0.6.0` | Spans: #490 (2026-08-09). `ScalingFactor` is planned for 0.9.0. Unreleased / partial. | capi all; core (`ScalingFactor`) | no |
| **P4-143** (filed from the #474 review; no own issue): `.cargo/config.toml` forces `+simd128`, which hid baseline-wasm32 build breaks from CI | n/a (CI configuration) | Other. No runtime defect in any published crate; it masked a *proposed* change that was backed out. | — | #554 (2026-08-13) | none | no |

### 2b. C ABI (`libjpeg-turbo-rs-capi`)

All rows below were introduced before the `v0.6.0` tag (2026-04-18/19 unless stated) and were therefore first published in capi 0.1.0. The vulnerable code was confirmed by grep in the published 0.1.0, 0.1.1 and 0.1.2 tarballs unless stated otherwise.

| Finding | Reachable from | Class | Introduced (commit, first tag) | Fixed (PR/commit, first tag) | Affected published versions | Advisory warranted? |
|---|---|---|---|---|---|---|
| **P4-125** (no GitHub issue found; filed from internal scan report `CLAUDE-SECURITY-20260807-214723`, F1/F2): `tj3DecompressToYUV8` copies 4 planes for a 4-component (CMYK/YCCK) JPEG into a `tj3YUVBufSize`-sized (3-plane) buffer. `tj3DecompressToYUVPlanes8` reads `dstPlanes[3]` past the caller's 3-entry array and writes a plane through it. | C ABI, documented-correct use. **The trigger is untrusted JPEG input.** | Heap buffer overflow (CWE-787) and a write through an out-of-bounds pointer | `4b296b6` 2026-04-18, `v0.6.0` | #450 (`ba89266`, 2026-08-09). Unreleased (main). | capi **0.1.0, 0.1.1, 0.1.2** | **yes**. This is the highest-severity C-ABI item. |
| **P4-195 / #615**: `jpeg_read_raw_data` / `jpeg12_read_raw_data` copy MCU-aligned plane widths and heights into caller rows that `libjpeg.txt` sizes to DCT blocks. That writes 8 bytes (16 for 12-bit) past each row for ordinary 4:2:0 widths, such as the manual's own 101×101 example. | C ABI, documented-correct use; the geometry comes from the JPEG | Heap buffer overflow (write) | `3027434` 2026-04-28, `v0.6.0` | **Open** | capi all | **yes** |
| **P4-165** (no GitHub issue found): with `TJSAMP_GRAY`, the YUV paths assume 3 planes. `tj3EncodeYUV8` writes 3 planes into a 1-plane `tj3YUVBufSize` buffer. `tj3CompressFromYUV8` and `tj3DecodeYUV8` read about 3× past it. The planar variants read past 1-element `planes`/`strides` arrays. | C ABI, documented-correct use | Heap overflow (write) and over-read | `4b296b6` + `3ea4a47` (GRAY→S444 map) 2026-04-18, `v0.6.0` | `887d6fe` in #558 (2026-08-14). Unreleased (main). | capi all | **yes** |
| **P4-145 / #514** (and the legacy half of P4-151 / #529): `TJPARAM_NOREALLOC`. (a) The in-place path of `tj3Compress8` ignored the declared `*jpegSize` capacity. (b) `tj3Compress12/16`, `tj3CompressFromYUV8`, `tj3CompressFromYUVPlanes8` and `tj3Transform` (plus legacy `tjCompress2`/`tjTransform` with `TJFLAG_NOREALLOC`) `free()` the caller's buffer and swap the pointer even when NOREALLOC is set. | C ABI. (b) is the documented NOREALLOC contract ("guarantees that it won't be [reallocated]"). (a) needs a buffer smaller than worst case whose real size is declared; upstream errors on that. | (a) heap overflow. (b) invalid free of a stack, static or `Vec` buffer; use-after-free or double free when the caller keeps its pointer. | `af52a38` (in-place path), `b81a683`, `4b296b6`, `42b7090`, 2026-04-18/19, `v0.6.0` | (a) #515 (`5bcf4ba`, 2026-08-10). (b) #532 (`ea770fb`, 2026-08-11). Unreleased (main). | capi all | **yes** |
| **P4-108 / #434**: `jpeg_mem_dest` with a caller-supplied buffer reads `*outsize` capacity as existing data and `free()`s the caller's pointer on the first flush. Stdio write errors are also ignored. | C ABI, documented-correct use (the libjpeg preallocated-buffer branch) | Invalid free (memory corruption). Garbage prefixed to the output. | `407e52c` 2026-04-19, `v0.6.0` | #443 (`c7ab145`, 2026-08-08). Unreleased (main). | capi all | **yes** |
| **P4-3 + P4-140 / #479 + P4-110** (no issue found for P4-3 or P4-110): v6b/v8 ABI confusion. capi 0.1.0 defaults SONAME to `libjpeg.so.62` (install name `libjpeg.62.dylib`) while exposing the v8 struct layout, and its crates.io description reads "drop-in replacement for libjpeg.so.62". `jpeg_Create*` ignore `version`/`struct_size`, so they write the full v8 mirror into a smaller v6b struct. | C ABI. A v6b-compiled consumer loading the library under its default name. In 0.1.1/0.1.2, only an explicit `CAPI_ACK_V6B_SONAME` build, but the crate-level docs (`src/lib.rs:5`) still offer `.so.62` substitution. | Memory corruption: writes past the caller's struct, wrong field offsets | SONAME `aeabcd0` 2026-04-18; Create `2606028` 2026-04-18; both `v0.6.0` | P4-3: `8a47439` 2026-05-17 (`v0.6.2`, published as capi **0.1.1**). P4-140 docs: #485 (2026-08-09). P4-110 guards: #525 (`868dc43`, 2026-08-11). Docs and guards are unreleased. | capi **0.1.0** (default build); capi 0.1.1–0.1.2 only for v6b opt-in builds that follow the stale crate doc | **yes** for 0.1.0; **consider** for 0.1.1–0.1.2 |
| **P4-196 / #616**: `jpeg_abort`, `jpeg_destroy` and the memmgr precision probe read `is_decompressor` at literal byte offset 32, which is correct on LP64 only | C ABI, documented calls, **32-bit (ILP32) builds only** | Type confusion: a compress object is torn down as a decompress object | `3027434` 2026-04-28, `v0.6.0` (the memmgr site was added later, before `v0.8.0`) | **Open** | capi all, on 32-bit targets | **yes** (32-bit only) |
| **P4-194 / #613**: memmgr `alloc_sarray`/`alloc_barray` multiply `JDIMENSION`s unchecked; upstream bounds and chunks | C ABI (vtable slots, called by consumers and modules with frame-derived sizes) | Heap overflow after a wrapped allocation (32-bit, with frame-derived sizes). On 64-bit, only through absurd direct arguments. | `b05a0be` 2026-04-19, `v0.6.0` | **Open** | capi all | **consider**. It becomes "yes" on 32-bit if a consumer sizes an array from frame dimensions (not demonstrated). |
| **P4-4** (no issue found): no `catch_unwind` on any `extern "C"` entry point. Any internal panic, such as the fuzz-found decoder panics (P4-74, P4-75, P4-161), unwinds to the FFI boundary. | C ABI, triggered by malformed input | Process abort. MSRV is 1.87 ≥ 1.81, where unwinding out of `extern "C"` aborts, so this is not UB. | capi inception, `v0.6.0` (0 `catch_unwind` in published 0.1.0) | `8a47439` 2026-05-17, `v0.6.2`, published as capi **0.1.1** (158 `unwind_guard!` uses) | capi **0.1.0** | **consider** |
| **P4-137 / #476**: `tj3Free`, `tj3Destroy` and most of the roughly 150 `extern "C"` exports (153 safe `extern "C"` fns in 0.1.0) are *safe* Rust `pub extern "C" fn`s (crate-wide `allow(clippy::not_unsafe_ptr_arg_deref)`). `handle_as_mut` forged an unbounded `&mut` lifetime. | Safe Rust, through the capi **rlib** only. C callers are not affected, because they always owned pointer validity. | UB from safe Rust (invalid or double free); unsound API | `e0b839c` / `6d7be1a` 2026-04-18, `v0.6.0` | #488 (2026-08-09), #508 (2026-08-10), #515 (2026-08-10). Unreleased (main). | capi all | **yes** (informational "unsound"). No Rust consumer of the rlib is known. |
| **P4-193 / #612**: `tj3Decompress8` forms `&mut [u8]` over `pitch * height`, while upstream touches only `pitch*(h-1) + row` | C ABI, only when the caller allocates tighter than upstream's "should normally be `pitch * destinationHeight`" | UB as a Rust validity violation (an over-long slice). No out-of-bounds access is performed. | `85c6b2a` 2026-04-18, `v0.6.0` | **Open** | capi all | **consider** (low) |
| **P4-164 gap 1**: after `jpeg_finish_decompress` / `jpeg_abort_decompress`, `next_input_byte`/`bytes_in_buffer` still point into the dropped owned source buffer | C ABI. The standard idiom `if (src->bytes_in_buffer == 0) fill()` after finish (multi-image streams). | Use-after-free read | **Unverified** for the published snapshots (note 9) | **Open** | likely capi all (unverified) | **consider**. It becomes "yes" once confirmed on a published snapshot. |

## 3. Notes on uncertain rows and how boundaries were set

1. **Method.** For each finding:
   - The "introduced" boundary is the commit where the vulnerable *code shape* became reachable from the public API. It was found with `git log -S` on the defining expression, then checked with `git tag --contains`.
   - The published version list then came from grepping that shape in each extracted crates.io tarball (`set_len` in `progressive_output.rs`, `cb_half.is_empty() && eff_row_height`, `fn prepare_ac_first_sse2`, `mem::forget` in `huffman_encode.rs`, `libc_free(prev_ptr)` in `jpeg_mem_dest`'s flush, `_version, _struct_size` unused in `jpeg_Create*`, `libc_free(prior)` in `precision.rs`/`yuv.rs`/`transform.rs`, no `MAX_YUV_PLANES`, `add(32)`).
   - Fix boundaries come from the PR merge plus `git tag --contains`. Every August fix is in no tag.
   - Introduced and fixed SHAs are from current local history. GitHub's recorded merge SHAs for PRs merged before the history rewrite differ: for example, #318 is `5bb4f8a` on GitHub and `9f54857` locally.

2. **P4-41: why published 0.5.0 is excluded.** The tag `v0.5.0` contains `c03bdbb`, but crates.io 0.5.0 was built from `954127e` (2026-04-11), which is not a descendant of `c03bdbb`. Grepping the 0.5.0 tarball finds no `cb_half`. The capi 0.1.0 row inherits P4-41 because its requirement `libjpeg-turbo-rs = "0.6"` can only resolve to 0.6.0–0.6.3, which are all affected.

3. **P4-135 scope by architecture.**
   - The x86_64 route is compile-verified (#481's probe, with no `unsafe` in the caller). It has two forms: out-of-bounds access through `width`, and AVX2 execution without AVX2.
   - Phase4 records the same safe-wrapper shape in the wasm32 and aarch64 modules, plus the `detect()` table route. The c620663 audit found unchecked lengths in `neon_fancy_h2v2_row` and four wasm encode wrappers.
   - I did not re-verify per architecture which of those were reachable from outside the crate in each published version. Treat non-x86_64 as "likely affected, unverified". The `simd::x86_64` module was gated on `target_arch` only, so `default-features = false` builds were also exposed.

4. **P4-136 severity.** The `set_len` contract violation is present in every published core, and uninitialized bytes reach safe code whenever a row is not written. That makes the API unsound regardless of exploitation. The "wrapped-small allocation the IDCT writes past" claim needs `comp_w * padded_h` (or `out_w * out_h * bpp`) to exceed `usize`:
   - On 64-bit this is impossible with 16-bit JPEG dimensions.
   - On 32-bit targets (wasm32, armv7, i686), `out_w*out_h*bpp` can wrap within the default `max_pixels`. However, the 128-byte-per-block coefficient buffers would probably fail or panic before `output()` is reached.

   I could not establish an end-to-end heap overflow without running it, so it is marked "unverified". The wasm npm package (`libjpeg-turbo-rs-wasm` 0.1.0–0.2.2) does not expose `ProgressiveDecoder`.

5. **P4-192 lower bound.** The custom-script API has existed since `v0.1.0` (`f96ea14`). The unchecked SSE2 kernel on that path arrived in `1f5e683`, and the published 0.4.0 tarball is the first to contain `fn prepare_ac_first_sse2`. In 0.3.0, `compress_progressive_custom` contains no `unsafe`, so an out-of-range `Se` there should panic rather than corrupt memory. I did not exhaustively audit every callee in 0.1.0–0.3.0.

6. **P4-138 reachability.** A double free needs `reserve` to *unwind* while the temporary `Vec` owns the buffer. With stable Rust's default allocation-error handling, an allocation failure aborts instead of unwinding. The only unwinding route is the `capacity overflow` panic (a request above `isize::MAX`):
   - It is unreachable on 64-bit from any encode.
   - It is theoretically reachable on 32-bit `panic=unwind` targets once the writer exceeds about 1 GiB.

   Phase4 confirmed the window only by calling the internal function directly under fault injection. Hence "consider", not "yes".

7. **P4-209 cluster.** Published cores have no fallible allocation at all (`try_reserve` count = 0 in every tarball), so the abort-on-refusal shape applies to every version. Whether a crafted header *actually* aborts depends on the platform: Linux overcommit with calloc-backed zero pages often defers failure to the OOM killer, while Windows, 32-bit targets and wasm fail immediately. capi 0.1.1+ catches panics but not allocation aborts, so the C ABI is affected in all capi versions.

8. **P4-3/P4-140/P4-110 for 0.1.0.** Two sources disagreed in 0.1.0. The crates.io description and default SONAME advertised `.so.62` substitution. The repository's `docs/ABI_COMPATIBILITY.md` and a build `cargo:warning` called that combination UB for v6b consumers. I count the shipped default and the registry description as the effective documentation, so the row is "yes" for 0.1.0. From 0.1.1, the default is `.so.8` and the remaining exposure is the stale crate-level doc sentence plus the missing struct-size guard. A v8 consumer compiled against matching headers is not affected in any version.

9. **P4-164 gap 1.** Phase4 says the dangling window "reproduces identically under the previous slurp implementation", which is the implementation that shipped through `v0.8.0` (P4-109 was reworked on 2026-08-14). I did not trace `jpeg_finish_decompress` in the 0.1.x tarballs, so the published-version column is "likely, unverified".

10. **Excluded after review, with reasons:**
    - P4-97: its aliasing UB was caught in review before merge and never shipped.
    - P4-149: an uninitialized `client_data` under `&mut`. The UCG question is unsettled, it was introduced by the unreleased P4-110 fix, and it was fixed before any release.
    - P4-148: test-only.
    - P4-126: phase4 says no out-of-bounds write is reachable.
    - P4-166: compresses the caller's own uninitialized buffer. The library makes no out-of-bounds access.
    - P4-184: a narrow pitch or stride is caller misuse.
    - The panic cluster (P4-74/P4-75 fixed 2026-07-30 and unreleased; P4-161 open): recoverable panics in Rust. In capi 0.1.0 they become aborts (the P4-4 row). In 0.1.1+, P4-161 surfaces as a silent zero-size success, which is a correctness issue, not a memory-safety one.

11. **P4-134 is not a soundness item.** On `main`, P4-134 is "No RISC-V RVV SIMD backend" (performance). The 2026-08-09 audit items were renumbered to P4-135..P4-141 after an ID collision (`5fc72c9`). Older notes that tie "P4-134" to the unsound-safe-API headline mean today's P4-135.

12. **Unclassified and needing a decision.**
    - P4-170: classic source-manager parity passes in debug and fails in `--release`. Phase4's open question is whether this is UB that the optimizer exposes. Until that is answered, it may belong in table 2b.
    - Also not assessed: the npm `libjpeg-turbo-rs-wasm` package beyond its API surface, and `git`-dependency users of the unpublished `v0.6.0` capi tree.

13. **Severity ordering for disclosure.**
    - The C-ABI rows driven by *file input* or *standard API use* (P4-125, P4-195, P4-165, P4-145(b), P4-108) carry the most real-world risk, and every published capi version is affected with no fixed release.
    - Among safe-Rust rows, P4-41 is the only one with a fixed published version (upgrade to ≥ 0.7.0 resolves it). P4-135, P4-136 and P4-192 affect every current release.
    - A patch release (core 0.8.1, capi 0.1.3) carrying the merged fixes would let advisories name a "patched" version. Today, every advisory except P4-41 would have to say "no patched version".
