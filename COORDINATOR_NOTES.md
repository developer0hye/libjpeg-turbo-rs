# Coordinator Notes — agent-aa781c88 (FFI B9-2 Pillow)

## FFI_B9_2_PILLOW — Pillow vs libjpeg-turbo-rs-capi shim

### Summary
Pillow (PIL) cannot drop-in-replace its libjpeg dependency with our
`libjpeg-turbo-rs-capi` cdylib today, because the crate exports **only**
the TurboJPEG (tj* / tj3*) API. Pillow's `_imaging.so` is compiled
against the **classic libjpeg API** (`jpeg_CreateCompress`,
`jpeg_read_header`, `jpeg_start_decompress`, `jpeg_finish_compress`,
`jpeg_std_error`, `jpeg_destroy_*`, etc.). That API surface is not
present in our cdylib.

This is a structural gap, not a bug — the B9 FFI milestone scoped the
TurboJPEG API only. B9-2 (this task) was the discovery probe.

### Evidence
- Exports in `crates/libjpeg-turbo-rs-capi/src/` (88 `extern "C" fn`
  symbols; all prefixed `tj*` / `tj3*`): `rg -n 'pub extern "C" fn'` →
  `alloc.rs:4 compress.rs:2 decompress.rs:2 header.rs:6 legacy.rs:36
  precision.rs:8 tj3.rs:12 transform.rs:2 yuv.rs:16`.
- `rg -n 'jpeg_CreateCompress|jpeg_CreateDecompress|jpeg_std_error|
  jpeg_read_header|jpeg_start_decompress|jpeg_start_compress'
  crates/libjpeg-turbo-rs-capi/src/` → **zero matches**.
- `build.rs` pins SONAME `libjpeg.so.62` / `@rpath/libjpeg.62.dylib`,
  which matches the classic libjpeg distribution name — so the library
  *advertises* itself as libjpeg.so.62, but *implements* only the
  TurboJPEG surface. The Pillow `_imaging.so` two-level-namespace link
  fails at dlopen time because the classic symbols are undefined.
- **Captured failure** (macOS 15, Pillow 12.2.0, Python 3.14, arm64):
  after `cp liblibjpeg_turbo_rs_capi.dylib
  PIL/.dylibs/libjpeg.62.4.0.dylib`, `import PIL.Image` raises:
  ```
  dlopen(.../PIL/_imaging.cpython-314-darwin.so, 0x0002):
    Symbol not found: _jpeg_resync_to_restart
    Referenced from: .../PIL/_imaging.cpython-314-darwin.so
    Expected in:     .../PIL/.dylibs/libjpeg.62.4.0.dylib  (our shim)
  ```
  Confirms (1) the shim loads as libjpeg.62.dylib at the OS level —
  SONAME/install_name plumbing works — and (2) `_imaging` needs the
  classic libjpeg API and finds none of it in our cdylib. First
  missing symbol surfaced by dyld is `jpeg_resync_to_restart`, but
  every other `jpeg_*` is equally absent (see phase-A log).
- The harness also verifies the **positive** half: `ctypes.CDLL(our_
  shim).tj3Init` resolves. TurboJPEG surface works; only the classic
  surface is missing.

### Artifacts produced
- `examples/pillow_smoke/test_pillow.py` — two-phase smoke test:
  - **Phase A:** `ctypes.CDLL(shim).tj3Init` probe + classic-symbol
    presence check. No side effects.
  - **Phase B:** Pillow decode → encode at q=90 → re-decode → PSNR.
    Asserts PSNR ≥ 30 dB (libjpeg-turbo's measured envelope is
    40–50 dB at q=90). Default fixture
    `tests/fixtures/cjpeg_240x320_portrait_444.jpg`, overridable via
    `PILLOW_SMOKE_FIXTURE`. Classifies `ImportError` on
    `from PIL import Image` as BLOCKER (code 3) — this is the exact
    path a classic-API symbol miss takes on macOS two-level-namespace
    linking.
  - Exit codes: 0 pass / 1 fail / 2 skip / 3 blocker.
- `examples/pillow_smoke/run.sh` — OS-aware driver:
  1. `cargo build -p libjpeg-turbo-rs-capi --release` if needed.
  2. Symlinks cdylib as `libjpeg.62.dylib` / `libjpeg.so.62` next to
     the original.
  3. Sets `DYLD_/LD_LIBRARY_PATH` for fresh dlopens.
  4. **Replaces Pillow's wheel-bundled libjpeg** (`PIL/.dylibs/
     libjpeg.62.4.0.dylib` on macOS, `PIL.libs/libjpeg-*.so.62.*`
     on manylinux) with a copy of our shim, with automatic
     `.pillow_smoke_backup` + `trap` restore on exit. Required
     because wheel-bundled libs use `@loader_path` / `$ORIGIN` install
     names that neither `DYLD_LIBRARY_PATH` nor `LD_LIBRARY_PATH` can
     override.
  5. Runs `test_pillow.py` and propagates its exit code.
- `tests/capi_pillow_compat.rs` — Rust integration test: spawns
  `bash run.sh`, maps `exit 3` to a documented SKIP pointing at this
  file, `exit 1` to a real panic, `exit 2` to SKIP, `exit 0` to pass.
  Skips cleanly on Windows and bash-less hosts.
- `tests/capi_pillow_compat.rs` — Rust integration test wrapper. Shells
  out to `run.sh`. Maps exit codes: 0 → pass, 2 → SKIP, 3 → SKIP with
  blocker message (documented here), 1 → panic. Windows and
  bash-less hosts skip.

### Pillow link model — which API does Pillow actually call?
Pillow links `libjpeg` (classic libjpeg-turbo-compatible API), **not**
`libturbojpeg`. Evidence: CPython's `_imaging` extension's source
(`src/libImaging/JpegEncode.c`, `JpegDecode.c`) uses `jpeg_std_error`,
`jpeg_create_compress`, `jpeg_CreateDecompress`, `jpeg_read_header`,
`jpeg_start_decompress`, etc. — all classic libjpeg symbols. The stock
Pillow-libjpeg-turbo wheels on PyPI ship a bundled `libjpeg.so.62` /
`libjpeg.62.dylib`. Therefore: **our shim must export the classic
libjpeg API — exporting only the TurboJPEG API is insufficient for
Pillow compatibility.** The task prompt's hint that "our shim exports
both" is not accurate for the current source tree.

### Decision
- The integration test `tests/capi_pillow_compat.rs` is wired to SKIP
  (with a precise reason string) when the shim returns exit code 3 —
  consistent with the project's "SKIP-with-reason acceptable for
  experimental discovery work" rule. It will flip to a real PASS/FAIL
  automatically once classic libjpeg symbols land.
- No `cargo fmt --check` / clippy issues: new code is formatter-clean,
  no warnings. See per-command output below.

### Follow-up work (for the coordinator / next agent)
1. Add a `crates/libjpeg-turbo-rs-capi/src/classic.rs` module exporting
   the classic libjpeg surface. Minimum-viable set for Pillow decode:
   `jpeg_std_error`, `jpeg_CreateDecompress` (macro → `jpeg_create_
   decompress`), `jpeg_mem_src`, `jpeg_read_header`, `jpeg_start_
   decompress`, `jpeg_read_scanlines`, `jpeg_finish_decompress`,
   `jpeg_destroy_decompress`. For encode add the `_compress` twins plus
   `jpeg_set_defaults`, `jpeg_set_quality`, `jpeg_mem_dest`,
   `jpeg_write_scanlines`.
2. The existing Rust encoder/decoder in the workspace root already
   implements every code path these C entry points need — this is a
   wrapping exercise, not algorithm work.
3. Re-run `tests/capi_pillow_compat.rs` once (1) lands; expect the test
   to flip from SKIP-blocker → PASS without modification.
4. Optional: also alias `libturbojpeg.so.0` → same cdylib via a second
   symlink once classic symbols exist, then both link paths Just Work
   as the `build.rs` header comment already claims.

### WORKSPACE_CARGO_ADDITIONS
(none — this task added no new dependencies. `test_pillow.py` uses
`PIL` + `numpy` inside a scratch venv managed by `run.sh`; these are
Python-side only and do not touch workspace `Cargo.toml`.)
