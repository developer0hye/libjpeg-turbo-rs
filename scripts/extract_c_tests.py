#!/usr/bin/env python3
"""
Extract EVERY test case from the C libjpeg-turbo test infrastructure
and output a structured inventory as JSON.

Parses:
  1. CMakeLists.txt          - add_test(), add_bittest(), MD5 variables
  2. test/tjcomptest.in      - lossy and lossless compression combos
  3. test/tjdecomptest.in    - decompression combos with crops and scales
  4. test/tjtrantest.in      - transform combos
  5. test/croptest.in        - crop iteration ranges
  6. src/tjunittest.c         - unit test functions, pixel formats, flags
"""

import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "references" / "libjpeg-turbo"


def read_file(relpath: str) -> str:
    full = BASE_DIR / relpath
    if not full.exists():
        print(f"WARNING: {full} not found", file=sys.stderr)
        return ""
    return full.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. Parse CMakeLists.txt
# ---------------------------------------------------------------------------
def parse_cmake() -> dict:
    text = read_file("CMakeLists.txt")

    # --- add_test(NAME ...) ---
    add_test_pat = re.compile(
        r"add_test\(\s*NAME\s+(\S+)\s+COMMAND\s+(.*?)\)",
        re.DOTALL,
    )
    cmake_tests = []
    for m in add_test_pat.finditer(text):
        name = m.group(1).strip()
        command = " ".join(m.group(2).split())
        cmake_tests.append({"name": name, "command": command})

    # --- add_bittest(...) ---
    # macro(add_bittest PROG NAME ARGS OUTFILE INFILE MD5SUM)
    add_bittest_pat = re.compile(
        r"add_bittest\(\s*(\$\{[^}]+\}|\S+)\s+(\S+)\s+"
        r'"([^"]*)"\s+'
        r"(\S+)\s+"
        r"(\S+)\s+"
        r"(\S+)",
        re.DOTALL,
    )
    bittests = []
    for m in add_bittest_pat.finditer(text):
        bittests.append({
            "program": m.group(1),
            "name": m.group(2),
            "args": m.group(3),
            "outfile": m.group(4),
            "infile": m.group(5),
            "md5": m.group(6),
        })

    # --- MD5 hash variables ---
    md5_pat = re.compile(r"set\((MD5_\w+)\s+([0-9a-fA-F]+)\)")
    md5_vars = {}
    for m in md5_pat.finditer(text):
        md5_vars[m.group(1)] = m.group(2)

    # Each add_bittest expands to 2 add_test calls (run + cmp), so total
    # cmake-registered tests = explicit add_test + 2 * add_bittest.
    # But we report the unique logical tests.
    return {
        "cmake_tests": cmake_tests,
        "cmake_add_test_count": len(cmake_tests),
        "cmake_bittests": bittests,
        "cmake_add_bittest_count": len(bittests),
        "cmake_total_test_registrations": len(cmake_tests) + 2 * len(bittests),
        "md5_hashes": {
            "count": len(md5_vars),
            "variables": sorted(md5_vars.keys()),
        },
    }


# ---------------------------------------------------------------------------
# 2. Parse tjcomptest.in
# ---------------------------------------------------------------------------
def parse_tjcomptest() -> dict:
    """Compute exact combo count by simulating the nested loops."""

    # --- Lossy section ---
    precisions = [8, 12]
    restartargs = ["", "-r 1 -icc $IMGDIR/test3.icc", "-r 1b"]
    ariargs = ["", "-a"]
    dctargs = ["", "-dc fa"]
    optargs = ["", "-o"]
    progargs = ["", "-p"]
    qualargs = ["", "-q 1", "-q 100"]
    subsamp_indices = list(range(6))  # 0..5
    # 4 image variants per subsamp iteration:
    #   rgb, grayscale-from-rgb, rgb-as-rgb, gray
    image_variants_per_subsamp = 4

    lossy_combos = 0
    for prec in precisions:
        for restartarg in restartargs:
            for ariarg in ariargs:
                for dctarg in dctargs:
                    for optarg in optargs:
                        # skip: optarg == "-o" and (ariarg == "-a" or prec == 12)
                        if optarg == "-o":
                            if ariarg == "-a" or prec == 12:
                                continue
                        for progarg in progargs:
                            # skip: progarg == "-p" and optarg == "-o"
                            if progarg == "-p" and optarg == "-o":
                                continue
                            for qualarg in qualargs:
                                for sampi in subsamp_indices:
                                    lossy_combos += image_variants_per_subsamp

    # --- Lossless section ---
    # for precision in {2..16}  => 15 values
    lossless_precisions = list(range(2, 17))
    # for psv in {1..7}  => 7
    psvs = list(range(1, 8))
    # for pt in {0..15}, skip if pt >= precision
    # for restartarg in "" "-r 1 -icc ..."  => 2
    lossless_restartargs = ["", "-r 1 -icc $IMGDIR/test3.icc"]
    # 2 image variants per iteration: rgb, gray
    lossless_images_per_iteration = 2

    lossless_combos = 0
    for prec in lossless_precisions:
        for psv in psvs:
            for pt in range(0, 16):
                if pt >= prec:
                    continue
                for restartarg in lossless_restartargs:
                    lossless_combos += lossless_images_per_iteration

    skip_conditions_lossy = [
        "optarg == '-o' and (ariarg == '-a' or precision == 12)",
        "progarg == '-p' and optarg == '-o'",
    ]
    skip_conditions_lossless = [
        "pt >= precision",
    ]

    return {
        "axes": {
            "precision": precisions,
            "restart": restartargs,
            "arithmetic": ariargs,
            "dct": dctargs,
            "optimize": optargs,
            "progressive": progargs,
            "quality": qualargs,
            "subsampling": ["444", "422", "440", "420", "411", "441"],
            "image_variants": [
                "rgb",
                "grayscale-from-rgb",
                "rgb-as-rgb",
                "gray",
            ],
        },
        "skip_conditions": skip_conditions_lossy,
        "estimated_lossy_combos": lossy_combos,
        "lossless_axes": {
            "precision": lossless_precisions,
            "psv": psvs,
            "pt_range": "0 to min(precision-1, 15)",
            "restart": lossless_restartargs,
            "images": ["rgb", "gray"],
        },
        "skip_conditions_lossless": skip_conditions_lossless,
        "estimated_lossless_combos": lossless_combos,
        "total_combos": lossy_combos + lossless_combos,
    }


# ---------------------------------------------------------------------------
# 3. Parse tjdecomptest.in
# ---------------------------------------------------------------------------
def parse_tjdecomptest() -> dict:
    """Simulate the nested loops exactly, counting combos."""

    precisions = [8, 12]
    subsamps = ["444", "422", "440", "420", "411", "441", "410", "gray"]
    cropargs = [
        "",
        "-cr 14x14+23+23",
        "-cr 21x21+4+4",
        "-cr 18x18+13+13",
        "-cr 21x21+0+0",
        "-cr 24x26+20+18",
    ]
    scaleargs = [
        "", "-s 16/8", "-s 15/8", "-s 14/8", "-s 13/8", "-s 12/8",
        "-s 11/8", "-s 10/8", "-s 9/8", "-s 7/8", "-s 6/8", "-s 5/8",
        "-s 4/8", "-s 3/8", "-s 2/8", "-s 1/8",
    ]
    nsargs = ["", "-nos"]
    dctargs = ["", "-dc fa"]

    lossy_combos = 0
    for prec in precisions:
        for subsamp in subsamps:
            for croparg in cropargs:
                # skip: croparg != "" and subsamp == "410"
                if croparg != "" and subsamp == "410":
                    continue
                for scalearg in scaleargs:
                    # skip: (scalearg in ["-s 1/8", "-s 2/8", "-s 3/8"]) and croparg != ""
                    if scalearg in ["-s 1/8", "-s 2/8", "-s 3/8"] and croparg != "":
                        continue
                    for nsarg in nsargs:
                        # skip: nsarg == "-nos" and subsamp not in [422, 420, 440]
                        if nsarg == "-nos" and subsamp not in ["422", "420", "440"]:
                            continue
                        for dctarg in dctargs:
                            # skip: dctarg == "-dc fa" and
                            #   (scalearg != "-s 4/8" or (subsamp != "420" and subsamp != "410"))
                            #   and scalearg != ""
                            if dctarg == "-dc fa":
                                cond1 = (
                                    scalearg != "-s 4/8"
                                    or (subsamp != "420" and subsamp != "410")
                                )
                                if cond1 and scalearg != "":
                                    continue
                            # Count image variants:
                            # gray: 2 (pgm + ppm via -r)
                            # non-gray: 2 if nsarg == "" (ppm + pgm via -g), 1 if nsarg == "-nos"
                            if subsamp == "gray":
                                variants = 2
                            else:
                                variants = 2 if nsarg == "" else 1
                            lossy_combos += variants

    # Lossless section: for precision in {2..16}: rgb + gray = 2 each
    lossless_precisions = list(range(2, 17))
    lossless_combos = len(lossless_precisions) * 2  # rgb + gray

    skip_conditions = [
        "croparg != '' and subsamp == '410'",
        "scalearg in ['1/8','2/8','3/8'] and croparg != ''",
        "nsarg == '-nos' and subsamp not in ['422','420','440']",
        "dctarg == '-dc fa' and (scalearg != '4/8' or subsamp not in ['420','410']) and scalearg != ''",
    ]

    crop_regions = [
        "14x14+23+23",
        "21x21+4+4",
        "18x18+13+13",
        "21x21+0+0",
        "24x26+20+18",
    ]

    scale_factors = [
        "16/8", "15/8", "14/8", "13/8", "12/8",
        "11/8", "10/8", "9/8", "7/8", "6/8", "5/8",
        "4/8", "3/8", "2/8", "1/8",
    ]

    return {
        "axes": {
            "precision": precisions,
            "subsampling": subsamps,
            "crop": ["(none)"] + crop_regions,
            "scale": ["(none)"] + scale_factors,
            "nosmooth": nsargs,
            "dct": dctargs,
            "output_variants": [
                "native-format",
                "grayscale-from-color (-g)",
                "rgb-from-gray (-r)",
            ],
        },
        "crop_regions": crop_regions,
        "scale_factors": scale_factors,
        "skip_conditions": skip_conditions,
        "estimated_lossy_combos": lossy_combos,
        "lossless_axes": {
            "precision": lossless_precisions,
            "images": ["rgb", "gray"],
        },
        "estimated_lossless_combos": lossless_combos,
        "total_combos": lossy_combos + lossless_combos,
    }


# ---------------------------------------------------------------------------
# 4. Parse tjtrantest.in
# ---------------------------------------------------------------------------
def parse_tjtrantest() -> dict:
    """Simulate the nested loops exactly."""

    precisions = [8, 12]
    subsamps = ["444", "422", "440", "420", "411", "441", "410", "gray"]
    ariargs = ["", "-a"]
    copyargs = ["", "-c i", "-c n"]
    cropargs = [
        "",
        "-cr 14x14+23+23",
        "-cr 21x21+4+4",
        "-cr 18x18+13+13",
        "-cr 21x21+0+0",
        "-cr 24x26+20+18",
    ]
    xformargs = [
        "", "-f h", "-f v", "-ro 90", "-ro 180", "-ro 270", "-t", "-transv",
    ]
    grayargs = ["", "-g"]
    optargs = ["", "-o"]
    progargs = ["", "-p"]
    restartargs = ["", "-r 1 -icc $IMGDIR/test3.icc", "-r 1b"]
    trimargs = ["", "-tri"]

    combos = 0
    for prec in precisions:
        for subsamp in subsamps:
            for ariarg in ariargs:
                for copyarg in copyargs:
                    # skip: copyarg == "-c n" and subsamp not in ["411","420"]
                    if copyarg == "-c n" and subsamp not in ["411", "420"]:
                        continue
                    # skip: copyarg == "-c i" and subsamp != "420"
                    if copyarg == "-c i" and subsamp != "420":
                        continue
                    for croparg in cropargs:
                        for xformarg in xformargs:
                            for grayarg in grayargs:
                                if grayarg == "":
                                    # skip: subsamp == "410" and croparg != ""
                                    if subsamp == "410" and croparg != "":
                                        continue
                                else:
                                    # skip: subsamp == "gray"
                                    if subsamp == "gray":
                                        continue
                                for optarg in optargs:
                                    # skip: optarg == "-o" and (ariarg == "-a" or prec == 12)
                                    if optarg == "-o":
                                        if ariarg == "-a" or prec == 12:
                                            continue
                                    for progarg in progargs:
                                        # skip: progarg == "-p" and optarg == "-o"
                                        if progarg == "-p" and optarg == "-o":
                                            continue
                                        for restartarg in restartargs:
                                            # skip: restartarg == "-r 1b" and croparg != ""
                                            if restartarg == "-r 1b" and croparg != "":
                                                continue
                                            for trimarg in trimargs:
                                                # skip: trimarg == "-tri" and (xformarg in ["-t",""] or croparg != "")
                                                if trimarg == "-tri":
                                                    if xformarg in ["-t", ""] or croparg != "":
                                                        continue
                                                combos += 1

    crop_regions = [
        "14x14+23+23",
        "21x21+4+4",
        "18x18+13+13",
        "21x21+0+0",
        "24x26+20+18",
    ]
    xform_types = [
        "(none)", "flip-horizontal", "flip-vertical",
        "rotate-90", "rotate-180", "rotate-270",
        "transpose", "transverse",
    ]

    skip_conditions = [
        "copyarg == '-c n' and subsamp not in ['411','420']",
        "copyarg == '-c i' and subsamp != '420'",
        "grayarg == '' and subsamp == '410' and croparg != ''",
        "grayarg == '-g' and subsamp == 'gray'",
        "optarg == '-o' and (ariarg == '-a' or precision == 12)",
        "progarg == '-p' and optarg == '-o'",
        "restartarg == '-r 1b' and croparg != ''",
        "trimarg == '-tri' and (xformarg in ['-t',''] or croparg != '')",
    ]

    return {
        "axes": {
            "precision": precisions,
            "subsampling": subsamps,
            "arithmetic": ariargs,
            "copy_mode": copyargs,
            "crop": ["(none)"] + crop_regions,
            "transform": xformargs,
            "grayscale": grayargs,
            "optimize": optargs,
            "progressive": progargs,
            "restart": restartargs,
            "trim": trimargs,
        },
        "crop_regions": crop_regions,
        "transform_types": xform_types,
        "copy_modes": ["(default)", "copy-icc", "copy-none"],
        "skip_conditions": skip_conditions,
        "estimated_combos": combos,
    }


# ---------------------------------------------------------------------------
# 5. Parse croptest.in
# ---------------------------------------------------------------------------
def parse_croptest() -> dict:
    """Simulate the croptest nested loops."""

    progargs = ["", "-progressive"]
    nsargs = ["", "-nosmooth"]
    colorsargs = ["", "-colors 256 -dither none -onepass"]
    y_range = list(range(0, 17))  # 0..16
    h_range = list(range(1, 17))  # 1..16
    samps = ["GRAY", "420", "422", "440", "444"]

    combos = 0
    for progarg in progargs:
        for nsarg in nsargs:
            for colorsarg in colorsargs:
                for y in y_range:
                    for h in h_range:
                        # Each iteration runs 5 samples (GRAY,420,422,440,444)
                        combos += len(samps)

    # Image dimensions from the script
    width = 128
    height = 95

    return {
        "axes": {
            "progressive": progargs,
            "nosmooth": nsargs,
            "colors": colorsargs,
            "Y_range": "0..16 (17 values)",
            "H_range": "1..16 (16 values)",
            "subsampling": samps,
        },
        "image": "vgl_6548_0026a.bmp",
        "image_dimensions": f"{width}x{height}",
        "crop_spec_formula": "W=WIDTH-X-7, X=(Y*16)%128, special case Y>15: Y2=HEIGHT-H",
        "Y_values": len(y_range),
        "H_values": len(h_range),
        "estimated_combos": combos,
    }


# ---------------------------------------------------------------------------
# 6. Parse tjunittest.c
# ---------------------------------------------------------------------------
def parse_tjunittest() -> dict:
    text = read_file("src/tjunittest.c")

    # Extract test function names
    func_pat = re.compile(r"^static\s+(?:int|void)\s+(\w+)\s*\(", re.MULTILINE)
    all_funcs = [m.group(1) for m in func_pat.finditer(text)]
    test_functions = [
        f for f in all_funcs
        if any(kw in f.lower() for kw in [
            "test", "comp", "decomp", "overflow", "buf", "bmp", "init",
            "check", "write",
        ])
    ]

    # Key test entry points called from main()
    main_test_functions = [
        "overflowTest",
        "doTest",
        "bufSizeTest",
        "bmpTest",
        "compTest",
        "decompTest",
        "_decompTest",
    ]

    pixel_formats = [
        "TJPF_RGB", "TJPF_BGR", "TJPF_RGBX", "TJPF_BGRX",
        "TJPF_XBGR", "TJPF_XRGB", "TJPF_GRAY",
        "TJPF_RGBA", "TJPF_BGRA", "TJPF_ABGR", "TJPF_ARGB",
        "TJPF_CMYK",
    ]

    pixel_format_groups = {
        "_3sampleFormats": ["TJPF_RGB", "TJPF_BGR"],
        "_4sampleFormats": ["TJPF_RGBX", "TJPF_BGRX", "TJPF_XBGR", "TJPF_XRGB", "TJPF_CMYK"],
        "_onlyGray": ["TJPF_GRAY"],
        "_onlyRGB": ["TJPF_RGB"],
    }

    subsampling_modes = [
        "TJSAMP_444", "TJSAMP_422", "TJSAMP_420", "TJSAMP_440",
        "TJSAMP_411", "TJSAMP_441", "TJSAMP_GRAY",
    ]

    flag_combinations = [
        "doYUV=0, lossless=0, alloc=0  (default lossy)",
        "doYUV=0, lossless=0, alloc=1  (lossy + alloc)",
        "doYUV=1, lossless=0, alloc=0  (yuv)",
        "doYUV=1, lossless=0, alloc=1  (yuv + alloc)",
        "doYUV=1, yuvAlign=1            (yuv nopad)",
        "doYUV=0, lossless=1, alloc=0  (lossless)",
        "doYUV=0, lossless=1, alloc=1  (lossless + alloc)",
        "bmp=1                          (bmp test)",
        "precision=12                   (12-bit)",
        "precision=12, alloc=1          (12-bit + alloc)",
        "precision=N (2-16), lossless=1 (N-bit lossless)",
        "precision=N (2-16), lossless=1, alloc=1",
        "precision=N (2-16), bmp=1      (N-bit bmp)",
    ]

    # Count doTest calls from main -- each doTest call does:
    #   for pfi in 0..nformats: for i in 0..1: compTest + decompTest
    #   If 4-sample format (pf >= RGBX and <= XRGB), also decompTest for RGBA variant
    # decompTest for non-lossless iterates over scaling factors
    # This is complex; we enumerate the doTest calls from main()

    do_test_calls_lossy = [
        # (w, h, formats_group, nformats, subsamp, basename)
        (35, 39, "_3sampleFormats", 2, "TJSAMP_444"),
        (39, 41, "_4sampleFormats", 5, "TJSAMP_444"),  # num4bf=5 (or 4 for yuv)
        (41, 35, "_3sampleFormats", 2, "TJSAMP_422"),
        # The following only run when !lossless:
        (35, 39, "_4sampleFormats", 5, "TJSAMP_422"),
        (39, 41, "_3sampleFormats", 2, "TJSAMP_420"),
        (41, 35, "_4sampleFormats", 5, "TJSAMP_420"),
        (35, 39, "_3sampleFormats", 2, "TJSAMP_440"),
        (39, 41, "_4sampleFormats", 5, "TJSAMP_440"),
        (41, 35, "_3sampleFormats", 2, "TJSAMP_411"),
        (35, 39, "_4sampleFormats", 5, "TJSAMP_411"),
        (39, 41, "_3sampleFormats", 2, "TJSAMP_441"),
        (41, 35, "_4sampleFormats", 5, "TJSAMP_441"),
    ]
    do_test_calls_always = [
        (35, 39, "_3sampleFormats", 2, "TJSAMP_444"),
        (39, 41, "_4sampleFormats", 5, "TJSAMP_444"),
        (41, 35, "_3sampleFormats", 2, "TJSAMP_422"),
    ]
    do_test_calls_gray = [
        (39, 41, "_onlyGray", 1, "TJSAMP_GRAY"),
    ]
    do_test_calls_gray_non_lossless = [
        (41, 35, "_3sampleFormats", 2, "TJSAMP_GRAY"),
        (35, 39, "_4sampleFormats", 4, "TJSAMP_GRAY"),  # num4bf=4 for gray
    ]
    do_test_calls_yuv = [
        (48, 48, "_onlyRGB", 1, "TJSAMP_444"),
        (48, 48, "_onlyRGB", 1, "TJSAMP_422"),
        (48, 48, "_onlyRGB", 1, "TJSAMP_420"),
        (48, 48, "_onlyRGB", 1, "TJSAMP_440"),
        (48, 48, "_onlyRGB", 1, "TJSAMP_411"),
        (48, 48, "_onlyRGB", 1, "TJSAMP_441"),
        (48, 48, "_onlyRGB", 1, "TJSAMP_GRAY"),
        (48, 48, "_onlyGray", 1, "TJSAMP_GRAY"),
    ]

    # bmpTest: align in [1,2,4,8] x format in [0..11] (TJ_NUMPF=12) x
    #          (bmp TD + ppm TD + bmp BU + ppm BU for prec==8;
    #           ppm TD + ppm BU for prec!=8)
    # Each doBmpTest also iterates targetPrecision from 2..16 (or 2..8 for bmp)
    bmp_aligns = [1, 2, 4, 8]
    bmp_formats = 12  # TJ_NUMPF
    bmp_combos_prec8 = len(bmp_aligns) * bmp_formats * 4  # bmp TD/BU + ppm TD/BU
    bmp_combos_other = len(bmp_aligns) * bmp_formats * 2  # ppm TD/BU only

    return {
        "test_functions": main_test_functions,
        "all_static_functions": test_functions,
        "pixel_formats": pixel_formats,
        "pixel_format_groups": pixel_format_groups,
        "subsampling_modes": subsampling_modes,
        "flag_combinations": flag_combinations,
        "doTest_calls": {
            "always_run": [
                f"doTest({w},{h},{fg},{n},{ss})"
                for w, h, fg, n, ss in do_test_calls_always
            ],
            "lossy_only": [
                f"doTest({w},{h},{fg},{n},{ss})"
                for w, h, fg, n, ss in do_test_calls_lossy[3:]
            ],
            "gray": [
                f"doTest({w},{h},{fg},{n},{ss})"
                for w, h, fg, n, ss in do_test_calls_gray
            ],
            "gray_non_lossless": [
                f"doTest({w},{h},{fg},{n},{ss})"
                for w, h, fg, n, ss in do_test_calls_gray_non_lossless
            ],
            "yuv_only": [
                f"doTest({w},{h},{fg},{n},{ss})"
                for w, h, fg, n, ss in do_test_calls_yuv
            ],
        },
        "bmpTest": {
            "aligns": bmp_aligns,
            "num_pixel_formats": bmp_formats,
            "combos_precision_8": bmp_combos_prec8,
            "combos_other_precision": bmp_combos_other,
        },
    }


# ---------------------------------------------------------------------------
# Assemble final inventory
# ---------------------------------------------------------------------------
def main():
    cmake = parse_cmake()
    tjcomp = parse_tjcomptest()
    tjdecomp = parse_tjdecomptest()
    tjtran = parse_tjtrantest()
    croptest = parse_croptest()
    tjunittest = parse_tjunittest()

    total_estimated = (
        tjcomp["total_combos"]
        + tjdecomp["total_combos"]
        + tjtran["estimated_combos"]
        + croptest["estimated_combos"]
        # cmake registered tests (each is one invocation)
        + cmake["cmake_total_test_registrations"]
    )

    inventory = {
        "cmake_tests": cmake["cmake_tests"],
        "cmake_add_test_count": cmake["cmake_add_test_count"],
        "cmake_bittests": cmake["cmake_bittests"],
        "cmake_add_bittest_count": cmake["cmake_add_bittest_count"],
        "cmake_total_test_registrations": cmake["cmake_total_test_registrations"],
        "tjcomptest": tjcomp,
        "tjdecomptest": tjdecomp,
        "tjtrantest": tjtran,
        "croptest": croptest,
        "tjunittest": tjunittest,
        "md5_hashes": cmake["md5_hashes"],
        "total_estimated_test_cases": total_estimated,
        "summary": {
            "cmake_registered_tests": cmake["cmake_total_test_registrations"],
            "tjcomptest_combos": tjcomp["total_combos"],
            "tjdecomptest_combos": tjdecomp["total_combos"],
            "tjtrantest_combos": tjtran["estimated_combos"],
            "croptest_combos": croptest["estimated_combos"],
            "grand_total": total_estimated,
        },
    }

    out_path = Path(__file__).resolve().parent / "c_test_inventory.json"
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  CMake add_test:        {cmake['cmake_add_test_count']}")
    print(f"  CMake add_bittest:     {cmake['cmake_add_bittest_count']}")
    print(f"  CMake total regs:      {cmake['cmake_total_test_registrations']}")
    print(f"  MD5 variables:         {cmake['md5_hashes']['count']}")
    print(f"  tjcomptest lossy:      {tjcomp['estimated_lossy_combos']}")
    print(f"  tjcomptest lossless:   {tjcomp['estimated_lossless_combos']}")
    print(f"  tjcomptest total:      {tjcomp['total_combos']}")
    print(f"  tjdecomptest lossy:    {tjdecomp['estimated_lossy_combos']}")
    print(f"  tjdecomptest lossless: {tjdecomp['estimated_lossless_combos']}")
    print(f"  tjdecomptest total:    {tjdecomp['total_combos']}")
    print(f"  tjtrantest:            {tjtran['estimated_combos']}")
    print(f"  croptest:              {croptest['estimated_combos']}")
    print(f"  GRAND TOTAL:           {total_estimated}")


if __name__ == "__main__":
    main()
