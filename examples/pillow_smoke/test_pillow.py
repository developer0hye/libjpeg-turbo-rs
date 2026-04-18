#!/usr/bin/env python3
"""FFI B9-2: Pillow (PIL) smoke test against libjpeg-turbo-rs-capi shim.

Two phases:
    (A) ctypes.CDLL-load our shim directly and probe for a TurboJPEG
        symbol (`tj3Init`). This proves the cdylib is loadable and the
        SONAME / install_name machinery works.
    (B) Force Pillow to use our shim by pre-replacing Pillow's bundled
        `libjpeg.62.dylib` / `libjpeg.so.62` with a symlink to our shim
        (run.sh takes care of the replacement before invoking us).
        Then decode + encode + round-trip at q=90, asserting PSNR >= 30 dB.

Exit codes:
    0  - success (both phases passed)
    2  - SKIP (Pillow / fixture / env not available)
    3  - BLOCKER (shim dlopen OK but Pillow aborts because classic libjpeg
                  API symbols like jpeg_CreateCompress are missing)
    1  - FAIL (shim loaded, Pillow loaded, but round-trip output wrong)
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import traceback
from pathlib import Path


def log(msg: str) -> None:
    print(f"[pillow_smoke] {msg}", flush=True)


def psnr(a, b) -> float:
    """Peak signal-to-noise ratio between two equal-shape numeric sequences
    of 0-255 samples. Returns +inf for pixel-identical inputs."""
    import numpy as np

    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if a_np.shape != b_np.shape:
        raise ValueError(f"shape mismatch: {a_np.shape} vs {b_np.shape}")
    mse = float(np.mean((a_np - b_np) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0**2) / mse)


def phase_a_dlopen_probe(shim_path: Path) -> int:
    """Direct ctypes.CDLL load of our shim + TurboJPEG symbol probe.

    This path does NOT depend on Pillow — if it fails, the cdylib itself
    is broken or has missing dependencies."""
    if not shim_path.is_file():
        log(f"SKIP-A: shim not found at {shim_path}")
        return 2
    try:
        lib = ctypes.CDLL(str(shim_path))
    except OSError as exc:
        log(f"BLOCKER-A: dlopen({shim_path}) failed: {exc}")
        return 3

    # Probe tj3Init (symbol presence only — no side-effects).
    try:
        tj3_init = lib.tj3Init
        tj3_init.argtypes = [ctypes.c_int]
        tj3_init.restype = ctypes.c_void_p
        log("phase-A: tj3Init symbol resolved in our shim")
    except AttributeError:
        log("BLOCKER-A: tj3Init not exported by our shim")
        return 3

    # Report whether classic libjpeg API symbols are present — they are
    # what Pillow needs. Absence is logged (not an immediate blocker here;
    # phase B will surface the fallout).
    classic_symbols = [
        "jpeg_CreateCompress",
        "jpeg_CreateDecompress",
        "jpeg_std_error",
        "jpeg_read_header",
        "jpeg_start_decompress",
        "jpeg_read_scanlines",
        "jpeg_finish_decompress",
        "jpeg_destroy_decompress",
    ]
    missing: list[str] = [s for s in classic_symbols if not hasattr(lib, s)]
    if missing:
        log("phase-A: classic libjpeg API symbols MISSING: " + ", ".join(missing))
        log(
            "phase-A: Pillow's _imaging.so uses the classic API — it will "
            "fail when bound against this shim."
        )
    else:
        log("phase-A: all probed classic libjpeg symbols present")
    return 0


def phase_b_pillow_roundtrip(fixture: Path) -> int:
    """Decode + re-encode + re-decode a fixture via Pillow and verify
    PSNR. If run.sh has already replaced Pillow's bundled libjpeg with
    our shim, this measures the true Pillow-on-rust-shim behaviour."""
    try:
        from PIL import Image, features
        import numpy as np  # noqa: F401
    except ModuleNotFoundError as exc:
        # Genuine "pip did not install Pillow/numpy" — skip.
        log(f"SKIP-B: required python module missing: {exc}")
        return 2
    except ImportError as exc:
        # Pillow is installed but _imaging.so failed to link — typically
        # because our shim replaced Pillow's bundled libjpeg and one or
        # more classic libjpeg symbols are undefined. This IS the
        # symbol-mismatch blocker we want to report.
        log(f"BLOCKER-B: Pillow _imaging failed to link against our shim: {exc}")
        return 3

    log(f"phase-B: Pillow version: {Image.__version__}")
    log(f"phase-B: features.libjpeg_turbo: {features.check_feature('libjpeg_turbo')}")

    if not fixture.is_file():
        log(f"SKIP-B: fixture not found: {fixture}")
        return 2

    log(f"phase-B: fixture: {fixture} ({fixture.stat().st_size} bytes)")

    try:
        with Image.open(fixture) as im:
            im.load()
            decoded = im.convert("RGB")
            w, h = decoded.size
            log(f"phase-B: decode OK: {w}x{h} mode={decoded.mode}")
    except Exception as exc:
        log(f"BLOCKER-B: Pillow failed to decode: {exc}")
        traceback.print_exc()
        return 3

    out_path: Path = fixture.parent / "pillow_smoke_roundtrip.jpg"
    try:
        decoded.save(out_path, "JPEG", quality=90)
        log(f"phase-B: encode OK: wrote {out_path} ({out_path.stat().st_size} bytes)")
    except Exception as exc:
        log(f"BLOCKER-B: Pillow failed to encode: {exc}")
        traceback.print_exc()
        return 3

    try:
        with Image.open(out_path) as im:
            im.load()
            rt = im.convert("RGB")
            if rt.size != decoded.size:
                log(f"FAIL: round-trip size changed: {decoded.size} -> {rt.size}")
                return 1
            peak = psnr(list(decoded.getdata()), list(rt.getdata()))
            log(f"phase-B: round-trip PSNR @ q=90: {peak:.2f} dB")
            if peak < 30.0:
                log(f"FAIL: PSNR {peak:.2f} dB below 30 dB threshold")
                return 1
    except Exception as exc:
        log(f"BLOCKER-B: Pillow failed to re-decode: {exc}")
        traceback.print_exc()
        return 3
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass

    log("phase-B PASS: decode + encode + round-trip PSNR >= 30 dB")
    return 0


def main() -> int:
    shim_env: str | None = os.environ.get("PILLOW_SMOKE_SHIM")
    if not shim_env:
        log("SKIP: PILLOW_SMOKE_SHIM env var not set")
        return 2
    shim_path: Path = Path(shim_env)

    fixture_env: str | None = os.environ.get("PILLOW_SMOKE_FIXTURE")
    if not fixture_env:
        log("SKIP: PILLOW_SMOKE_FIXTURE env var not set")
        return 2
    fixture: Path = Path(fixture_env)

    rc_a: int = phase_a_dlopen_probe(shim_path)
    if rc_a != 0:
        return rc_a

    rc_b: int = phase_b_pillow_roundtrip(fixture)
    return rc_b


if __name__ == "__main__":
    sys.exit(main())
