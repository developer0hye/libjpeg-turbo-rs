#!/usr/bin/env python3
"""Seed corpora for the C-oracle differential targets.

usage: python3 fuzz/scripts/make_c_oracle_diff_seeds.py <cjpeg 3.x> [corpus dir]

Writes, under the corpus dir (default fuzz/corpus):
- fuzz_raw_diff_c/       baseline, progressive and restart streams over the
                         sampling layouts cjpeg writes, including unusual ones
- fuzz_ycbcr_diff_c/     the colour ones, each behind a few option prefixes
                         (scale, box/fancy upsampling, crop; see the target)
- fuzz_lossless_diff_c/  SOF3 streams: every predictor, a point transform,
  and fuzz_decompress_precision/  precisions 2/8/12/16, restart intervals
                         (one relabelled 1x3, which libjpeg mis-decodes; the
                         lossless target hands it libjpeg as 1x1), plus
                         layouts cjpeg cannot write (non-interleaved scans,
                         subsampled frames), from the encoder below

Every stream is a legal JPEG that libjpeg-turbo 3.2.0 decodes. Re-running
overwrites the same file names.
"""
import os
import subprocess
import sys
import tempfile

W, H = 61, 45  # odd, so every layout has partial MCUs


def pnm(channels, maxval):
    """A textured image: gradients plus fine detail, so interpolated and
    replicated chroma differ."""
    out = bytearray()
    for y in range(H):
        for x in range(W):
            values = [
                (x * maxval // (W - 1) + (x * y) % 7 * maxval // 28) % (maxval + 1),
                (y * maxval // (H - 1) + (x + y) % 5 * maxval // 23) % (maxval + 1),
                ((x + 2 * y) * maxval // 150 + (x * 3) % 13 * maxval // 36) % (maxval + 1),
            ][:channels]
            for v in values:
                out += v.to_bytes(2, "big") if maxval > 255 else bytes([v])
    magic = b"P6" if channels == 3 else b"P5"
    return magic + b"\n%d %d\n%d\n" % (W, H, maxval) + bytes(out)


def cjpeg(tool, image, args):
    with tempfile.NamedTemporaryFile(suffix=".pnm", delete=False) as f:
        f.write(image)
    try:
        return subprocess.run([tool, *args, f.name], check=True, capture_output=True).stdout
    finally:
        os.unlink(f.name)


# ---- T.81 Process 14 encoder for layouts cjpeg cannot write --------------
# Predictor 1, point transform 0, 8-bit, RGB-coded (Adobe transform 0).

BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # T.81 Table K.3
VALS = list(range(12))


def huff_codes():
    codes, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(BITS[length - 1]):
            codes[VALS[k]] = (code, length)
            code += 1
            k += 1
        code <<= 1
    return codes


CODES = huff_codes()


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def put(self, value, length):
        for i in range(length - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0x00)
                self.acc, self.n = 0, 0

    def flush(self):
        while self.n:
            self.put(1, 1)
        return bytes(self.out)


def encode_diff(bw, diff):
    ssss = abs(diff).bit_length()
    code, length = CODES[ssss]
    bw.put(code, length)
    if ssss:
        bw.put(diff if diff >= 0 else diff + (1 << ssss) - 1, ssss)


def segment(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def p14(width, height, sampling, scans, seed):
    """sampling: [(h, v)] per component; scans: lists of component indices."""
    hmax = max(h for h, _ in sampling)
    vmax = max(v for _, v in sampling)
    planes, dims = [], []
    for ci, (h, v) in enumerate(sampling):
        cw, ch = width * h // hmax, height * v // vmax
        dims.append((cw, ch))
        planes.append([[(x * (37 + 11 * ci) + y * (53 + 7 * ci) + (x * y + seed * (ci + 1)) % 29 * 5) % 256
                        for x in range(cw)] for y in range(ch)])

    def predict(plane, x, y):
        if x == 0 and y == 0:
            return 128
        if y == 0 or x != 0:
            return plane[y][x - 1]
        return plane[y - 1][x]

    out = bytearray(b"\xff\xd8")
    out += segment(0xEE, b"Adobe" + bytes([0, 100, 0, 0, 0, 0, 0]))
    ids = [ord("R"), ord("G"), ord("B")]
    sof = bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([len(sampling)])
    for ci, (h, v) in enumerate(sampling):
        sof += bytes([ids[ci], (h << 4) | v, 0])
    out += segment(0xC3, sof)
    out += segment(0xC4, bytes([0x00]) + bytes(BITS) + bytes(VALS))
    for scan in scans:
        sos = bytes([len(scan)])
        for ci in scan:
            sos += bytes([ids[ci], 0x00])
        out += segment(0xDA, sos + bytes([1, 0, 0]))
        bw = BitWriter()
        if len(scan) == 1:
            ci = scan[0]
            cw, ch = dims[ci]
            for y in range(ch):
                for x in range(cw):
                    encode_diff(bw, planes[ci][y][x] - predict(planes[ci], x, y))
        else:
            for my in range(height // vmax):
                for mx in range(width // hmax):
                    for ci in scan:
                        h, v = sampling[ci]
                        for dy in range(v):
                            for dx in range(h):
                                x, y = mx * h + dx, my * v + dy
                                encode_diff(bw, planes[ci][y][x] - predict(planes[ci], x, y))
        out += bw.flush()
    return bytes(out + b"\xff\xd9")


SYNTHETIC_LOSSLESS = {
    "three_scans": ([(1, 1)] * 3, [[0], [1], [2]]),
    "one_then_two_scans": ([(1, 1)] * 3, [[0], [1, 2]]),
    "two_then_one_scans": ([(1, 1)] * 3, [[0, 1], [2]]),
    "s422": ([(2, 1), (1, 1), (1, 1)], [[0, 1, 2]]),
    "s420": ([(2, 2), (1, 1), (1, 1)], [[0, 1, 2]]),
    "s440": ([(1, 2), (1, 1), (1, 1)], [[0, 1, 2]]),
    "s420_three_scans": ([(2, 2), (1, 1), (1, 1)], [[0], [1], [2]]),
}

# ---- corpora ---------------------------------------------------------------

LAYOUTS = ["1x1", "2x1", "2x2", "1x2", "4x1", "4x2", "3x1",
           "2x2,2x1,1x1", "2x2,1x2,2x1", "4x1,2x1,1x1", "1x1,2x2,2x2"]
MODES = {"baseline": [], "progressive": ["-progressive"], "restart": ["-restart", "1"]}
# (options byte, crop x/width, crop y/height): see fuzz_ycbcr_diff_c.rs.
PREFIXES = [
    (8, 0x00, 0x00),           # 1/1, fancy
    (8 | 0x10, 0x00, 0x00),    # 1/1, box
    (12, 0x00, 0x00),          # 1/2, fancy
    (13 | 0x20, 0x73, 0x52),   # 3/8, crop
    (8 | 0x20, 0x00, 0x31),    # 1/1, crop from the left edge, 3 output columns wide
    (0, 0x00, 0x00),           # 2/1, fancy
]


def main():
    tool = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "fuzz/corpus"
    dirs = {name: os.path.join(root, name) for name in
            ["fuzz_raw_diff_c", "fuzz_ycbcr_diff_c", "fuzz_lossless_diff_c", "fuzz_decompress_precision"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def write(target, name, data):
        with open(os.path.join(dirs[target], name), "wb") as f:
            f.write(data)

    rgb, grey = pnm(3, 255), pnm(1, 255)
    count = {name: 0 for name in dirs}
    for layout in LAYOUTS:
        for mode, args in MODES.items():
            jpeg = cjpeg(tool, rgb, ["-quality", "90", "-sample", layout, *args])
            tag = f"diffseed_{layout.replace(',', '_')}_{mode}"
            write("fuzz_raw_diff_c", tag + ".jpg", jpeg)
            count["fuzz_raw_diff_c"] += 1
            for k, prefix in enumerate(PREFIXES):
                write("fuzz_ycbcr_diff_c", f"{tag}_opt{k}.bin", bytes(prefix) + jpeg)
                count["fuzz_ycbcr_diff_c"] += 1
    for mode, args in MODES.items():
        write("fuzz_raw_diff_c", f"diffseed_grey_{mode}.jpg",
              cjpeg(tool, grey, ["-quality", "90", "-grayscale", *args]))
        count["fuzz_raw_diff_c"] += 1

    lossless = {}
    for psv in range(1, 8):
        lossless[f"rgb_p8_psv{psv}"] = cjpeg(tool, rgb, ["-lossless", str(psv)])
    lossless["rgb_p8_psv1_pt2"] = cjpeg(tool, rgb, ["-lossless", "1,2"])
    lossless["rgb_p8_restart1"] = cjpeg(tool, rgb, ["-lossless", "1", "-restart", "1"])
    lossless["grey_p8_psv6"] = cjpeg(tool, grey, ["-lossless", "6"])
    lossless["grey_p2_psv1"] = cjpeg(tool, pnm(1, 3), ["-lossless", "1", "-precision", "2"])
    lossless["grey_p12_psv1"] = cjpeg(tool, pnm(1, 4095), ["-lossless", "1", "-precision", "12"])
    lossless["rgb_p12_psv7_pt3"] = cjpeg(tool, pnm(3, 4095), ["-lossless", "7,3", "-precision", "12"])
    lossless["grey_p16_psv4"] = cjpeg(tool, pnm(1, 65535), ["-lossless", "4", "-precision", "16"])
    lossless["grey_p2_psv1_restart2"] = cjpeg(
        tool, pnm(1, 3), ["-lossless", "1", "-precision", "2", "-restart", "2"])
    lossless["grey_p12_psv7_restart1"] = cjpeg(
        tool, pnm(1, 4095), ["-lossless", "7", "-precision", "12", "-restart", "1"])
    lossless["rgb_p16_psv4_restart3"] = cjpeg(
        tool, pnm(3, 65535), ["-lossless", "4", "-precision", "16", "-restart", "3"])
    # The lone component relabelled 1x3 (T.81 leaves a one-component scan
    # unchanged); a 2-row interval is not a multiple of 3.
    relabelled = bytearray(cjpeg(tool, grey, ["-lossless", "1", "-restart", "2"]))
    relabelled[relabelled.index(b"\xff\xc3") + 4 + 7] = 0x13
    lossless["grey_p8_psv1_restart2_sampling13"] = bytes(relabelled)
    for seed, (name, (sampling, scans)) in enumerate(SYNTHETIC_LOSSLESS.items()):
        lossless[f"synthetic_{name}"] = p14(16, 8, sampling, scans, seed)
    for name, jpeg in lossless.items():
        for target in ("fuzz_lossless_diff_c", "fuzz_decompress_precision"):
            write(target, f"diffseed_lossless_{name}.jpg", jpeg)
            count[target] += 1
    for name, n in count.items():
        print(f"{name}: {n} seeds")


if __name__ == "__main__":
    main()
