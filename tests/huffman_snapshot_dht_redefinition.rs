//! Explicit regression for the per-scan Huffman-table snapshot invariant
//! (issue #351 acceptance criterion): a DHT emitted *after* a scan's SOS
//! must never retroactively alter the tables that earlier scan decodes
//! against.
//!
//! Real-world shape: C `cjpeg -optimize` with a non-interleaved sequential
//! scan script emits a fresh optimized DHT before *every* scan. The Cb and
//! Cr scans both use table slot 1, so the Cr scan's DHT **redefines** the
//! slot the Cb scan already used. A decoder that resolves tables from
//! final marker state (instead of a per-scan snapshot) decodes Cb with
//! Cr's tables and diverges from djpeg.
//!
//! Skip rule: missing C tools soft-skip locally but hard-fail in CI so the
//! gate cannot silently vanish.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn is_ci() -> bool {
    std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
}

fn tool_path(name: &str) -> Option<PathBuf> {
    for dir in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/libjpeg-turbo/bin",
    ] {
        let pb = PathBuf::from(dir).join(name);
        if pb.exists() {
            return Some(pb);
        }
    }
    which(name)
}

fn which(name: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(name).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8(out.stdout).ok()?;
    let path = path.trim();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

/// Photo-like 64×64 RGB content: smooth gradients plus texture so every
/// component has non-trivial AC statistics (distinct optimized tables).
fn test_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            let r = (x * 4) as u8;
            let g = (y * 4) as u8;
            let b = (((x * 7) ^ (y * 13)) & 0xff) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

fn write_ppm(path: &std::path::Path, pixels: &[u8], width: usize, height: usize) {
    let mut f = std::fs::File::create(path).expect("create ppm");
    write!(f, "P6\n{width} {height}\n255\n").expect("ppm header");
    f.write_all(pixels).expect("ppm body");
}

/// Parse the raw PPM produced by `djpeg` into (width, height, rgb bytes).
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut fields = Vec::new();
    let mut pos = 0usize;
    while fields.len() < 4 && pos < data.len() {
        while pos < data.len() && data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if data[pos] == b'#' {
            while pos < data.len() && data[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < data.len() && !data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(std::str::from_utf8(&data[start..pos]).unwrap().to_string());
    }
    assert_eq!(fields[0], "P6", "djpeg must emit binary PPM");
    let width: usize = fields[1].parse().unwrap();
    let height: usize = fields[2].parse().unwrap();
    pos += 1; // single whitespace after maxval
    (width, height, data[pos..].to_vec())
}

/// Walk JPEG markers, returning the payload of every DHT segment that
/// defines the given (class, slot), in stream order.
fn dht_definitions(jpeg: &[u8], class: u8, slot: u8) -> Vec<Vec<u8>> {
    let mut defs = Vec::new();
    let mut i = 2usize; // skip SOI
    while i + 3 < jpeg.len() {
        if jpeg[i] != 0xFF {
            i += 1;
            continue;
        }
        let code = jpeg[i + 1];
        if code == 0xFF {
            i += 1;
            continue;
        }
        // Standalone markers.
        if code == 0xD8 || code == 0xD9 || code == 0x01 || (0xD0..=0xD7).contains(&code) {
            i += 2;
            continue;
        }
        let len = ((jpeg[i + 2] as usize) << 8) | jpeg[i + 3] as usize;
        let seg_start = i + 4;
        let seg_end = i + 2 + len;
        if code == 0xC4 {
            // DHT may contain several tables back to back.
            let mut p = seg_start;
            while p < seg_end {
                let tc_th = jpeg[p];
                let bits = &jpeg[p + 1..p + 17];
                let nsyms: usize = bits.iter().map(|&b| b as usize).sum();
                let table_end = p + 17 + nsyms;
                if (tc_th >> 4) == class && (tc_th & 0x0F) == slot {
                    defs.push(jpeg[p..table_end].to_vec());
                }
                p = table_end;
            }
        }
        if code == 0xDA {
            // Walk entropy data to the next real marker.
            let mut k = seg_end;
            while k + 1 < jpeg.len() {
                if jpeg[k] == 0xFF {
                    let nxt = jpeg[k + 1];
                    if nxt == 0x00 || (0xD0..=0xD7).contains(&nxt) {
                        k += 2;
                    } else {
                        break;
                    }
                } else {
                    k += 1;
                }
            }
            i = k;
            continue;
        }
        i = seg_end;
    }
    defs
}

#[test]
fn late_dht_redefinition_does_not_alter_earlier_scan() {
    let (Some(cjpeg), Some(djpeg)) = (tool_path("cjpeg"), tool_path("djpeg")) else {
        if is_ci() {
            panic!("cjpeg/djpeg must be installed in CI — the DHT-snapshot gate cannot skip");
        }
        eprintln!("SKIP: cjpeg/djpeg not found");
        return;
    };

    let dir = std::env::temp_dir().join(format!("dht_redef_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ppm_path = dir.join("input.ppm");
    let scans_path = dir.join("scans.txt");
    let pixels = test_pixels();
    write_ppm(&ppm_path, &pixels, 64, 64);
    // Non-interleaved sequential: one full scan (Ss=0..Se=63) per component.
    std::fs::write(&scans_path, "0: 0 63 0 0;\n1: 0 63 0 0;\n2: 0 63 0 0;\n")
        .expect("scans script");

    let cjpeg_out = Command::new(&cjpeg)
        .args(["-optimize", "-sample", "1x1", "-quality", "90", "-scans"])
        .arg(&scans_path)
        .arg(&ppm_path)
        .output()
        .expect("run cjpeg");
    assert!(
        cjpeg_out.status.success(),
        "cjpeg failed: {}",
        String::from_utf8_lossy(&cjpeg_out.stderr)
    );
    let jpeg = cjpeg_out.stdout;

    // Precondition: the stream must actually redefine DC/AC slot 1 between
    // the Cb and Cr scans with different table contents. If cjpeg's
    // emission strategy ever changes, fail loudly instead of testing
    // nothing.
    for class in [0u8, 1u8] {
        let defs = dht_definitions(&jpeg, class, 1);
        assert!(
            defs.len() >= 2,
            "expected ≥2 DHT definitions of class {class} slot 1, got {}",
            defs.len()
        );
        assert!(
            defs.windows(2).any(|w| w[0] != w[1]),
            "expected differing redefinitions of class {class} slot 1 — \
             identical tables would not exercise the snapshot invariant"
        );
    }

    // Ours.
    let mut decoder = libjpeg_turbo_rs::Decoder::new(&jpeg).expect("parse");
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    let ours = decoder.decode_image().expect("decode");
    assert_eq!((ours.width, ours.height), (64, 64));

    // djpeg reference.
    let mut child = Command::new(&djpeg)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn djpeg");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&jpeg)
        .expect("feed djpeg");
    let djpeg_out = child.wait_with_output().expect("djpeg output");
    assert!(
        djpeg_out.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&djpeg_out.stderr)
    );
    let (w, h, reference) = parse_ppm(&djpeg_out.stdout);
    assert_eq!((w, h), (64, 64));
    // zip() would silently truncate to the shorter side; a short djpeg
    // raster must fail, not pass with max_diff == 0.
    assert_eq!(ours.data.len(), reference.len());

    // Byte-exact agreement: any use of final-state tables for an earlier
    // scan shows up as a large diff in the Cb channel.
    let max_diff: i32 = ours
        .data
        .iter()
        .zip(reference.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "decode must be byte-exact vs djpeg on per-scan DHT redefinition"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
