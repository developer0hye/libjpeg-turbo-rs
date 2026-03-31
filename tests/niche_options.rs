use libjpeg_turbo_rs::{
    decompress, Encoder, MarkerStreamWriter, PixelFormat, SavedMarker, Subsampling,
};

fn gradient_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255) / width.max(1)) as u8);
            pixels.push(((y * 255) / height.max(1)) as u8);
            pixels.push(128);
        }
    }
    pixels
}

#[test]
fn smoothing_factor_zero_produces_valid_jpeg() {
    let pixels = gradient_pixels(32, 32);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(0)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn smoothing_factor_produces_valid_jpeg() {
    let pixels = gradient_pixels(32, 32);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(50)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn smoothing_factor_changes_output() {
    let mut pixels = vec![0u8; 32 * 32 * 3];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 37 + 13) % 256) as u8;
    }
    let no_smooth = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(0)
        .encode()
        .unwrap();
    let with_smooth = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(50)
        .smoothing_factor(100)
        .encode()
        .unwrap();
    assert_ne!(no_smooth, with_smooth);
}

#[test]
fn fancy_downsampling_on_vs_off_produces_different_output() {
    // Use noisy pixels with high-frequency chroma detail to make the
    // triangle pre-filter visibly different from box-only downsampling.
    let mut pixels = vec![0u8; 64 * 64 * 3];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i * 37 + i / 3 * 53 + 7) % 256) as u8;
    }
    let fancy = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .fancy_downsampling(true)
        .encode()
        .unwrap();
    let simple = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .fancy_downsampling(false)
        .encode()
        .unwrap();
    assert_ne!(fancy, simple);
    assert_eq!(decompress(&fancy).unwrap().width, 64);
    assert_eq!(decompress(&simple).unwrap().width, 64);
}

#[test]
fn fancy_downsampling_default_is_true() {
    let pixels = gradient_pixels(32, 32);
    let default_enc = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .encode()
        .unwrap();
    let fancy_enc = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .fancy_downsampling(true)
        .encode()
        .unwrap();
    assert_eq!(default_enc, fancy_enc);
}

#[test]
fn jfif_version_override_reflected_in_output() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .jfif_version(1, 2)
        .encode()
        .unwrap();
    assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    assert_eq!(&jpeg[2..4], &[0xFF, 0xE0]);
    assert_eq!(&jpeg[6..11], b"JFIF\0");
    assert_eq!(jpeg[11], 1);
    assert_eq!(jpeg[12], 2);
}

#[test]
fn jfif_default_version_is_1_01() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    assert_eq!(jpeg[11], 1);
    assert_eq!(jpeg[12], 1);
}

fn find_marker(data: &[u8], code: u8) -> Option<usize> {
    for i in 0..data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] == code {
            return Some(i);
        }
    }
    None
}

#[test]
fn adobe_marker_toggle_for_cmyk() {
    let pixels = vec![128u8; 16 * 16 * 4];
    let with_adobe = Encoder::new(&pixels, 16, 16, PixelFormat::Cmyk)
        .quality(75)
        .encode()
        .unwrap();
    assert!(
        find_marker(&with_adobe, 0xEE).is_some(),
        "CMYK should include Adobe APP14 by default"
    );
    let without_adobe = Encoder::new(&pixels, 16, 16, PixelFormat::Cmyk)
        .quality(75)
        .write_adobe_marker(false)
        .encode()
        .unwrap();
    assert!(
        find_marker(&without_adobe, 0xEE).is_none(),
        "Adobe APP14 should be absent when disabled"
    );
}

#[test]
fn adobe_marker_explicit_enable_for_non_cmyk() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let normal = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    assert!(
        find_marker(&normal, 0xEE).is_none(),
        "RGB should not include Adobe APP14 by default"
    );
    let with_adobe = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .write_adobe_marker(true)
        .encode()
        .unwrap();
    assert!(
        find_marker(&with_adobe, 0xEE).is_some(),
        "Adobe APP14 should be present when explicitly enabled"
    );
}

#[test]
fn marker_stream_writer_produces_valid_segment() {
    let mut writer = MarkerStreamWriter::new(0xE5);
    writer.write_byte(0x01);
    writer.write_byte(0x02);
    writer.write_bytes(&[0x03, 0x04, 0x05]);
    let segment = writer.finish();
    assert_eq!(segment[0], 0xFF);
    assert_eq!(segment[1], 0xE5);
    assert_eq!(u16::from_be_bytes([segment[2], segment[3]]), 7);
    assert_eq!(&segment[4..9], &[0x01, 0x02, 0x03, 0x04, 0x05]);
    assert_eq!(segment.len(), 9);
}

#[test]
fn marker_stream_writer_empty_data() {
    let writer = MarkerStreamWriter::new(0xE1);
    let segment = writer.finish();
    assert_eq!(&segment[0..2], &[0xFF, 0xE1]);
    assert_eq!(u16::from_be_bytes([segment[2], segment[3]]), 2);
    assert_eq!(segment.len(), 4);
}

#[test]
fn custom_marker_processor_receives_data() {
    use libjpeg_turbo_rs::SavedMarker;
    use std::sync::{Arc, Mutex};
    let pixels = vec![128u8; 8 * 8 * 3];
    let marker_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .saved_marker(SavedMarker {
            code: 0xE5,
            data: marker_data.clone(),
        })
        .encode()
        .unwrap();
    let received: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let received_clone = Arc::clone(&received);
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_marker_processor(0xE5, move |data: &[u8]| -> Option<Vec<u8>> {
        *received_clone.lock().unwrap() = Some(data.to_vec());
        Some(data.to_vec())
    });
    let _img = decoder.decode_image().unwrap();
    let received_data = received.lock().unwrap().take();
    assert!(
        received_data.is_some(),
        "marker processor should have been called"
    );
    assert_eq!(received_data.unwrap(), marker_data);
}

// -----------------------------------------------------------------------
// C djpeg cross-validation for niche encoder options
// -----------------------------------------------------------------------

/// Path to C djpeg binary, or `None` if not installed.
fn djpeg_path() -> Option<std::path::PathBuf> {
    let homebrew: std::path::PathBuf = std::path::PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    std::process::Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| std::path::PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn niche_parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = niche_ppm_skip_ws(data, idx);
    let (width, next) = niche_ppm_read_num(data, idx);
    idx = niche_ppm_skip_ws(data, next);
    let (height, next) = niche_ppm_read_num(data, idx);
    idx = niche_ppm_skip_ws(data, next);
    let (_maxval, next) = niche_ppm_read_num(data, idx);
    idx = next + 1;
    let pixels: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixels.len(),
        width * height * 3,
        "PPM pixel data length mismatch"
    );
    (width, height, pixels)
}

fn niche_ppm_skip_ws(data: &[u8], mut idx: usize) -> usize {
    loop {
        while idx < data.len() && data[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    idx
}

fn niche_ppm_read_num(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

struct NicheTempFile {
    path: std::path::PathBuf,
}

impl NicheTempFile {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!(
                "ljt_niche_{}_{}_{name}",
                std::process::id(),
                id
            )),
        }
    }
}

impl Drop for NicheTempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Encode a JPEG, decode with both C djpeg and Rust, assert pixel-identical output.
fn assert_djpeg_matches_rust(
    djpeg: &std::path::Path,
    jpeg: &[u8],
    width: usize,
    height: usize,
    label: &str,
) {
    let tmp_jpg = NicheTempFile::new(&format!("{label}.jpg"));
    let tmp_ppm = NicheTempFile::new(&format!("{label}.ppm"));

    std::fs::write(&tmp_jpg.path, jpeg).expect("write temp JPEG");

    let output = std::process::Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm.path)
        .arg(&tmp_jpg.path)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed for {label}: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm.path).expect("read PPM");
    let (c_w, c_h, c_pixels) = niche_parse_ppm(&ppm_data);
    assert_eq!(c_w, width, "{label}: C djpeg width mismatch");
    assert_eq!(c_h, height, "{label}: C djpeg height mismatch");

    let rust_image = decompress(jpeg).expect("Rust decompress failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "{label}: pixel data length mismatch"
    );

    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  {label}: pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
    }
    assert_eq!(
        max_diff, 0,
        "{label}: {} pixels differ, max_diff={}",
        mismatch_count, max_diff
    );
}

#[test]
fn c_djpeg_cross_validation_niche_options() {
    let djpeg = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = gradient_pixels(width, height);

    // (a) Encode with Adobe APP14 marker explicitly enabled for RGB
    {
        let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(90)
            .subsampling(Subsampling::S444)
            .write_adobe_marker(true)
            .encode()
            .expect("encode with Adobe marker failed");

        // Verify Adobe APP14 marker is present
        assert!(
            find_marker(&jpeg, 0xEE).is_some(),
            "Adobe APP14 marker should be present"
        );

        assert_djpeg_matches_rust(&djpeg, &jpeg, width, height, "adobe_app14");
    }

    // (b) Encode with custom JFIF version (1, 2)
    {
        let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(90)
            .subsampling(Subsampling::S444)
            .jfif_version(1, 2)
            .encode()
            .expect("encode with custom JFIF version failed");

        // Verify JFIF version is 1.02
        assert_eq!(jpeg[11], 1, "JFIF major version");
        assert_eq!(jpeg[12], 2, "JFIF minor version");

        assert_djpeg_matches_rust(&djpeg, &jpeg, width, height, "jfif_version");
    }

    // (c) Encode with custom APP marker (APP5 with arbitrary data)
    {
        let marker_data: Vec<u8> = vec![0xCA, 0xFE, 0xBA, 0xBE, 0x01, 0x02, 0x03];
        let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(90)
            .subsampling(Subsampling::S444)
            .saved_marker(SavedMarker {
                code: 0xE5,
                data: marker_data,
            })
            .encode()
            .expect("encode with custom marker failed");

        // Verify APP5 marker is present
        assert!(
            find_marker(&jpeg, 0xE5).is_some(),
            "APP5 marker should be present"
        );

        assert_djpeg_matches_rust(&djpeg, &jpeg, width, height, "custom_app_marker");
    }
}
