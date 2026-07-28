/// Generate a diverse JPEG test corpus in tests/corpus/.
///
/// Usage: cargo run --example generate_corpus
///
/// Output layout:
///   tests/corpus/generated/  — JPEGs produced by C cjpeg from synthetic/reference PPMs
///   tests/corpus/fuzz_seeds/ — copies of fuzz/corpus/fuzz_decompress/*.jpg
///   tests/corpus/fixtures/   — copies of tests/fixtures/*.jpg
use std::io::Read;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// C tool helpers
// ---------------------------------------------------------------------------

fn c_tool_path(name: &str) -> Option<PathBuf> {
    let homebrew = PathBuf::from(format!("/opt/homebrew/bin/{}", name));
    if homebrew.exists() {
        return Some(homebrew);
    }
    std::process::Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn run_cjpeg(cjpeg: &Path, input_ppm: &Path, output_jpg: &Path, args: &[&str]) -> bool {
    let output = std::process::Command::new(cjpeg)
        .args(args)
        .arg("-outfile")
        .arg(output_jpg)
        .arg(input_ppm)
        .output();
    match output {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            eprintln!("cjpeg failed: {}", String::from_utf8_lossy(&o.stderr));
            false
        }
        Err(e) => {
            eprintln!("cjpeg error: {}", e);
            false
        }
    }
}

// ---------------------------------------------------------------------------
// PPM generation (P6 binary format)
// ---------------------------------------------------------------------------

fn write_ppm(path: &Path, width: usize, height: usize, pixels: &[u8]) {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut data = Vec::with_capacity(header.len() + pixels.len());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(pixels);
    std::fs::write(path, &data).expect("failed to write PPM");
}

fn make_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 255) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

fn make_solid(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    vec![r, g, b]
        .into_iter()
        .cycle()
        .take(width * height * 3)
        .collect()
}

fn make_checkerboard(width: usize, height: usize, block: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let is_white = ((x / block) + (y / block)).is_multiple_of(2);
            let v = if is_white { 255u8 } else { 0u8 };
            pixels.push(v);
            pixels.push(v);
            pixels.push(v);
        }
    }
    pixels
}

/// Deterministic pseudo-random noise (LCG-based, no external deps)
fn make_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    let mut state = seed;
    for _ in 0..(width * height * 3) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        pixels.push((state >> 33) as u8);
    }
    pixels
}

/// High-contrast edges (vertical and horizontal bars with color transitions)
fn make_edges(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let h_edge = (x % 16 < 8) as u8 * 255;
            let v_edge = (y % 16 < 8) as u8 * 255;
            let diag = ((x + y) % 8 < 4) as u8 * 200;
            pixels.push(h_edge);
            pixels.push(v_edge);
            pixels.push(diag);
        }
    }
    pixels
}

/// Sine wave pattern (smooth gradients with varying frequency)
fn make_sine(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64 / width.max(1) as f64;
            let fy = y as f64 / height.max(1) as f64;
            let r = ((fx * 4.0 * std::f64::consts::PI).sin() * 127.0 + 128.0) as u8;
            let g = ((fy * 6.0 * std::f64::consts::PI).sin() * 127.0 + 128.0) as u8;
            let b = (((fx + fy) * 3.0 * std::f64::consts::PI).sin() * 127.0 + 128.0) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Mixed-frequency pattern simulating natural-looking content
fn make_natural(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    let mut state: u64 = 42;
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64 / width.max(1) as f64;
            let fy = y as f64 / height.max(1) as f64;
            // Low-frequency base
            let base_r = (fx * 2.0 * std::f64::consts::PI).sin() * 60.0 + 128.0;
            let base_g = (fy * 1.5 * std::f64::consts::PI).cos() * 60.0 + 128.0;
            let base_b = ((fx * fy) * 4.0 * std::f64::consts::PI).sin() * 40.0 + 128.0;
            // High-frequency noise
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f64 / 255.0 - 0.5) * 30.0;
            let r = (base_r + noise).clamp(0.0, 255.0) as u8;
            let g = (base_g + noise * 0.7).clamp(0.0, 255.0) as u8;
            let b = (base_b + noise * 0.5).clamp(0.0, 255.0) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// Source PPM descriptors
// ---------------------------------------------------------------------------

struct SourcePpm {
    name: String,
    path: PathBuf,
}

fn prepare_sources(tmp_dir: &Path) -> Vec<SourcePpm> {
    let mut sources: Vec<SourcePpm> = Vec::new();

    // Reference PPMs from libjpeg-turbo testimages
    for name in &["testorig", "testimg"] {
        let path = PathBuf::from(format!("references/libjpeg-turbo/testimages/{}.ppm", name));
        if path.exists() {
            sources.push(SourcePpm {
                name: name.to_string(),
                path,
            });
        }
    }

    if sources.is_empty() {
        eprintln!("warning: no reference PPMs found in references/libjpeg-turbo/testimages/");
    }

    // Synthetic images: (name, width, height, pixel_fn)
    type PixelFn = Box<dyn Fn(usize, usize) -> Vec<u8>>;
    let mut synthetics: Vec<(&str, usize, usize, PixelFn)> = vec![
        ("gradient_64x64", 64, 64, Box::new(make_gradient)),
        ("gradient_640x480", 640, 480, Box::new(make_gradient)),
        (
            "solid_red_8x8",
            8,
            8,
            Box::new(|w, h| make_solid(w, h, 255, 0, 0)),
        ),
        (
            "checkerboard_32x32",
            32,
            32,
            Box::new(|w, h| make_checkerboard(w, h, 2)),
        ),
        ("tiny_1x1", 1, 1, Box::new(make_gradient)),
        ("tiny_3x3", 3, 3, Box::new(make_gradient)),
        ("odd_7x11", 7, 11, Box::new(make_gradient)),
        ("odd_33x17", 33, 17, Box::new(make_gradient)),
        ("strip_100x1", 100, 1, Box::new(make_gradient)),
        ("strip_1x100", 1, 100, Box::new(make_gradient)),
        // Phase 2 additions: more resolutions, content patterns, edge cases
        (
            "noise_320x240",
            320,
            240,
            Box::new(|w, h| make_noise(w, h, 12345)),
        ),
        (
            "noise_17x31",
            17,
            31,
            Box::new(|w, h| make_noise(w, h, 67890)),
        ),
        ("edges_256x256", 256, 256, Box::new(make_edges)),
        ("sine_800x600", 800, 600, Box::new(make_sine)),
        ("natural_640x480", 640, 480, Box::new(make_natural)),
        ("natural_1280x720", 1280, 720, Box::new(make_natural)),
        ("odd_127x63", 127, 63, Box::new(make_gradient)),
        (
            "odd_255x127",
            255,
            127,
            Box::new(|w, h| make_noise(w, h, 99999)),
        ),
        ("tall_16x256", 16, 256, Box::new(make_gradient)),
        ("wide_256x16", 256, 16, Box::new(make_edges)),
        (
            "solid_black_8x8",
            8,
            8,
            Box::new(|w, h| make_solid(w, h, 0, 0, 0)),
        ),
        (
            "solid_white_8x8",
            8,
            8,
            Box::new(|w, h| make_solid(w, h, 255, 255, 255)),
        ),
        // Large resolutions: 4K (always included)
        ("natural_3840x2160", 3840, 2160, Box::new(make_natural)),
    ];

    // 8K sources are opt-in via CORPUS_INCLUDE_8K=1 (too slow for CI)
    if std::env::var("CORPUS_INCLUDE_8K").as_deref() == Ok("1") {
        synthetics.push(("gradient_7680x4320", 7680, 4320, Box::new(make_gradient)));
        synthetics.push((
            "noise_7680x4320",
            7680,
            4320,
            Box::new(|w, h| make_noise(w, h, 77777)),
        ));
        synthetics.push(("edges_7680x4320", 7680, 4320, Box::new(make_edges)));
        // Odd 8K dimensions (non-MCU-aligned)
        synthetics.push(("natural_7681x4321", 7681, 4321, Box::new(make_natural)));
    }

    for (name, width, height, gen) in &synthetics {
        let ppm_path = tmp_dir.join(format!("{}.ppm", name));
        let pixels = gen(*width, *height);
        write_ppm(&ppm_path, *width, *height, &pixels);
        sources.push(SourcePpm {
            name: name.to_string(),
            path: ppm_path,
        });
    }

    sources
}

// ---------------------------------------------------------------------------
// cjpeg variant matrix
// ---------------------------------------------------------------------------

struct CjpegVariant {
    /// Used in the output filename: {source}_{label}.jpg
    label: String,
    args: Vec<String>,
}

fn all_variants() -> Vec<CjpegVariant> {
    let mut variants: Vec<CjpegVariant> = Vec::new();

    // Axes
    let subsampling: &[(&str, Option<&str>)] = &[
        ("420", Some("2x2")),
        ("422", Some("2x1")),
        ("444", Some("1x1")),
        ("440", Some("1x2")),
        ("411", Some("4x1")),
        ("gray", None), // uses -grayscale instead
    ];
    let qualities: &[u32] = &[1, 10, 25, 50, 75, 90, 95, 100];

    // Flags that get OR'd with the base baseline pass
    // (label suffix, extra args)
    let flag_sets: &[(&str, &[&str])] = &[
        ("baseline", &[]),
        ("progressive", &["-progressive"]),
        ("optimized", &["-optimize"]),
        ("arithmetic", &["-arithmetic"]),
        ("arithmetic_progressive", &["-arithmetic", "-progressive"]),
        ("restart1", &["-restart", "1"]),
        ("restart4", &["-restart", "4"]),
        ("smooth50", &["-smooth", "50"]),
    ];

    for (subsamp_label, sample_arg) in subsampling {
        for &quality in qualities {
            for (flag_label, extra_args) in flag_sets {
                // Arithmetic + progressive is only meaningful at some qualities; generate for all.
                let label = format!("{}_{}_{}", subsamp_label, quality, flag_label);
                let mut args: Vec<String> = Vec::new();

                if let Some(sample) = sample_arg {
                    args.push("-sample".to_string());
                    args.push(sample.to_string());
                } else {
                    args.push("-grayscale".to_string());
                }

                args.push("-quality".to_string());
                args.push(quality.to_string());

                for a in *extra_args {
                    args.push(a.to_string());
                }

                variants.push(CjpegVariant { label, args });
            }
        }
    }

    variants
}

fn generate_jpegs(cjpeg: &Path, sources: &[SourcePpm], out_dir: &Path) -> (usize, usize) {
    let variants = all_variants();

    // Collect all (source, variant) pairs for parallel execution
    let work: Vec<(&SourcePpm, &CjpegVariant)> = sources
        .iter()
        .flat_map(|source| variants.iter().map(move |variant| (source, variant)))
        .collect();

    let generated = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::atomic::AtomicUsize::new(0);

    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk_size = work.len().div_ceil(parallelism);

    let generated_ref = &generated;
    let failed_ref = &failed;

    std::thread::scope(|s| {
        for chunk in work.chunks(chunk_size) {
            s.spawn(move || {
                for (source, variant) in chunk {
                    let filename = format!("{}_{}.jpg", source.name, variant.label);
                    let output_jpg = out_dir.join(&filename);
                    let args: Vec<&str> = variant.args.iter().map(|s| s.as_str()).collect();
                    if run_cjpeg(cjpeg, &source.path, &output_jpg, &args) {
                        generated_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        failed_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });

    (
        generated.load(std::sync::atomic::Ordering::Relaxed),
        failed.load(std::sync::atomic::Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// File copying
// ---------------------------------------------------------------------------

fn copy_jpgs(src_dir: &Path, dst_dir: &Path) -> Result<usize, String> {
    copy_jpgs_recursive(src_dir, src_dir, dst_dir)
}

fn is_jpeg_candidate(path: &Path) -> Result<bool, String> {
    let extension_is_jpeg = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
        });
    if extension_is_jpeg {
        return Ok(true);
    }
    if path.extension().is_some() {
        return Ok(false);
    }

    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("open corpus input {}: {error}", path.display()))?;
    let mut signature = [0; 2];
    let bytes_read = file
        .read(&mut signature)
        .map_err(|error| format!("read corpus input {}: {error}", path.display()))?;
    Ok(bytes_read == signature.len() && signature == [0xff, 0xd8])
}

fn copy_selected_jpgs(root: &Path, paths: &[PathBuf], dst_dir: &Path) -> Result<usize, String> {
    let mut count = 0;
    for path in paths {
        let metadata = std::fs::symlink_metadata(path)
            .map_err(|error| format!("inspect corpus path {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!(
                "tracked corpus input must be a regular file: {}",
                path.display()
            ));
        }
        if !is_jpeg_candidate(path)? {
            continue;
        }
        let relative = path.strip_prefix(root).map_err(|error| {
            format!(
                "derive relative corpus path for {} from {}: {error}",
                path.display(),
                root.display()
            )
        })?;
        let destination = dst_dir.join(relative);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!("create corpus destination {}: {error}", parent.display())
            })?;
        }
        std::fs::copy(path, &destination).map_err(|error| {
            format!(
                "copy corpus input {} to {}: {error}",
                path.display(),
                destination.display()
            )
        })?;
        count += 1;
    }
    Ok(count)
}

fn copy_git_tracked_jpgs(root: &Path, dst_dir: &Path) -> Result<usize, String> {
    let output = std::process::Command::new("git")
        .args(["ls-files", "-z", "--"])
        .arg(root)
        .output()
        .map_err(|error| format!("list tracked corpus inputs: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let paths = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            std::str::from_utf8(path)
                .map(PathBuf::from)
                .map_err(|error| format!("tracked corpus path is not UTF-8: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    copy_selected_jpgs(root, &paths, dst_dir)
}

fn copy_jpgs_recursive(root: &Path, current_dir: &Path, dst_dir: &Path) -> Result<usize, String> {
    let mut count: usize = 0;
    let entries = std::fs::read_dir(current_dir)
        .map_err(|error| format!("read corpus directory {}: {error}", current_dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "read corpus directory entry under {}: {error}",
                current_dir.display()
            )
        })?;
        let path: PathBuf = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("inspect corpus path {}: {error}", path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "symlink is not allowed in corpus: {}",
                path.display()
            ));
        }
        if file_type.is_dir() {
            count += copy_jpgs_recursive(root, &path, dst_dir)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }

        if is_jpeg_candidate(&path)? {
            let relative: &Path = path.strip_prefix(root).map_err(|error| {
                format!(
                    "derive relative corpus path for {} from {}: {error}",
                    path.display(),
                    root.display()
                )
            })?;
            let dst: PathBuf = dst_dir.join(relative);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent).map_err(|error| {
                    format!("create corpus destination {}: {error}", parent.display())
                })?;
            }
            std::fs::copy(&path, &dst).map_err(|error| {
                format!(
                    "copy corpus input {} to {}: {error}",
                    path.display(),
                    dst.display()
                )
            })?;
            count += 1;
        }
    }
    Ok(count)
}

fn retain_matching_files<F>(current_dir: &Path, accept: &mut F) -> Result<(usize, usize), String>
where
    F: FnMut(&Path) -> Result<bool, String>,
{
    let mut inspected = 0;
    let mut retained = 0;
    let entries = std::fs::read_dir(current_dir)
        .map_err(|error| format!("read corpus directory {}: {error}", current_dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "read corpus directory entry under {}: {error}",
                current_dir.display()
            )
        })?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("inspect corpus path {}: {error}", path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "symlink is not allowed in corpus: {}",
                path.display()
            ));
        }
        if file_type.is_dir() {
            let (child_inspected, child_retained) = retain_matching_files(&path, accept)?;
            inspected += child_inspected;
            retained += child_retained;
        } else if file_type.is_file() {
            inspected += 1;
            if accept(&path)? {
                retained += 1;
            } else {
                std::fs::remove_file(&path).map_err(|error| {
                    format!("remove rejected parity input {}: {error}", path.display())
                })?;
            }
        }
    }
    Ok((inspected, retained))
}

fn strict_djpeg_accepts(djpeg: &Path, jpeg: &Path) -> Result<bool, String> {
    let output = std::process::Command::new(djpeg)
        .args(["-strict", "-outfile", "/dev/null"])
        .arg(jpeg)
        .output()
        .map_err(|error| format!("run strict djpeg for {}: {error}", jpeg.display()))?;
    Ok(output.status.success())
}

fn assert_bucket_minimums(
    generated_count: usize,
    fuzz_seed_source_count: usize,
    fuzz_seed_parity_count: usize,
    fixture_count: usize,
) -> Result<(), String> {
    const MIN_GENERATED: usize = 9_000;
    const MIN_FUZZ_SEED_SOURCES: usize = 1_100;
    const MIN_FUZZ_SEED_PARITY: usize = 300;
    const MIN_FIXTURES: usize = 180;

    for (name, actual, minimum) in [
        ("generated", generated_count, MIN_GENERATED),
        (
            "fuzz_seed_sources",
            fuzz_seed_source_count,
            MIN_FUZZ_SEED_SOURCES,
        ),
        (
            "fuzz_seed_parity",
            fuzz_seed_parity_count,
            MIN_FUZZ_SEED_PARITY,
        ),
        ("fixtures", fixture_count, MIN_FIXTURES),
    ] {
        if actual < minimum {
            return Err(format!(
                "{name} corpus bucket has {actual} files, below required minimum {minimum}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{assert_bucket_minimums, copy_jpgs, copy_selected_jpgs, retain_matching_files};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempTree {
        path: PathBuf,
    }

    impl TempTree {
        fn new(label: &str) -> Self {
            let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path: PathBuf = std::env::temp_dir().join(format!(
                "libjpeg_corpus_copy_{}_{}_{}",
                std::process::id(),
                counter,
                label
            ));
            std::fs::create_dir_all(&path).expect("create temp tree");
            Self { path }
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).ok();
        }
    }

    #[test]
    fn copy_jpgs_preserves_nested_paths_and_extensionless_jpeg_seeds() {
        let source: TempTree = TempTree::new("source");
        let destination: TempTree = TempTree::new("destination");
        let nested: PathBuf = source.path.join("real_world/camera");
        std::fs::create_dir_all(&nested).expect("create nested source");

        std::fs::write(source.path.join("top.jpg"), b"not validated here")
            .expect("write top-level JPEG");
        std::fs::write(nested.join("photo.jpeg"), b"nested fixture").expect("write nested JPEG");
        std::fs::write(nested.join("00592c456b03"), b"\xff\xd8\xff\xd9")
            .expect("write extensionless JPEG seed");
        std::fs::write(nested.join("not-a-jpeg"), b"plain corpus bytes")
            .expect("write non-JPEG corpus input");

        let copied: usize = copy_jpgs(&source.path, &destination.path).expect("copy corpus");

        assert_eq!(copied, 3, "all and only JPEG candidates must be copied");
        assert!(destination.path.join("top.jpg").is_file());
        assert!(
            destination
                .path
                .join("real_world/camera/photo.jpeg")
                .is_file(),
            "nested relative path must be preserved"
        );
        assert!(
            destination
                .path
                .join("real_world/camera/00592c456b03")
                .is_file(),
            "extensionless SOI seed must be included"
        );
        assert!(!destination
            .path
            .join("real_world/camera/not-a-jpeg")
            .exists());
    }

    #[test]
    fn selected_copy_excludes_unlisted_corpus_files() {
        let source = TempTree::new("selected-source");
        let destination = TempTree::new("selected-destination");
        let tracked = source.path.join("tracked.jpg");
        let untracked = source.path.join("untracked.jpg");
        std::fs::write(&tracked, b"tracked").expect("write selected input");
        std::fs::write(&untracked, b"untracked").expect("write unselected input");

        let copied = copy_selected_jpgs(&source.path, &[tracked], &destination.path)
            .expect("copy selected corpus files");

        assert_eq!(copied, 1);
        assert!(destination.path.join("tracked.jpg").is_file());
        assert!(!destination.path.join("untracked.jpg").exists());
    }

    #[test]
    fn copy_jpgs_fails_when_source_cannot_be_read() {
        let source: TempTree = TempTree::new("missing-source");
        let destination: TempTree = TempTree::new("missing-destination");
        std::fs::remove_dir_all(&source.path).expect("remove source");

        let error: String = copy_jpgs(&source.path, &destination.path)
            .expect_err("missing source must fail closed");

        assert!(error.contains("read corpus directory"), "{error}");
        assert!(
            error.contains(&source.path.display().to_string()),
            "{error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn copy_jpgs_rejects_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let source: TempTree = TempTree::new("symlink-source");
        let destination: TempTree = TempTree::new("symlink-destination");
        symlink(&source.path, source.path.join("cycle")).expect("create symlink cycle");

        let error: String = copy_jpgs(&source.path, &destination.path)
            .expect_err("symlinked corpus paths must be rejected");

        assert!(error.contains("symlink"), "{error}");
        assert!(error.contains("cycle"), "{error}");
    }

    #[test]
    fn source_bucket_minimums_reject_silently_shrunken_corpora() {
        assert!(assert_bucket_minimums(9_000, 1_100, 300, 180).is_ok());

        for counts in [
            (8_999, 1_100, 300, 180),
            (9_000, 1_099, 300, 180),
            (9_000, 1_100, 299, 180),
            (9_000, 1_100, 300, 179),
        ] {
            let error = assert_bucket_minimums(counts.0, counts.1, counts.2, counts.3)
                .expect_err("shrunken source bucket must fail closed");
            assert!(error.contains("below required minimum"), "{error}");
        }
    }

    #[test]
    fn parity_filter_removes_rejected_inputs_and_reports_both_counts() {
        let root = TempTree::new("filter");
        std::fs::create_dir_all(root.path.join("nested")).expect("create filter fixture");
        std::fs::write(root.path.join("accepted"), b"accepted").expect("write accepted input");
        std::fs::write(root.path.join("nested/rejected"), b"rejected")
            .expect("write rejected input");

        let (inspected, retained) = retain_matching_files(&root.path, &mut |path| {
            Ok(path.file_name().and_then(|name| name.to_str()) == Some("accepted"))
        })
        .expect("filter parity inputs");

        assert_eq!((inspected, retained), (2, 1));
        assert!(root.path.join("accepted").is_file());
        assert!(!root.path.join("nested/rejected").exists());
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cjpeg = match c_tool_path("cjpeg") {
        Some(p) => p,
        None => {
            eprintln!(
                "warning: cjpeg not found in /opt/homebrew/bin or PATH — cannot generate JPEG corpus"
            );
            eprintln!("install libjpeg-turbo (e.g. `brew install libjpeg-turbo`) and re-run.");
            std::process::exit(1);
        }
    };
    let djpeg = match c_tool_path("djpeg") {
        Some(p) => p,
        None => {
            eprintln!("warning: djpeg not found — cannot validate fuzz parity inputs");
            std::process::exit(1);
        }
    };
    println!("cjpeg: {}", cjpeg.display());
    println!("djpeg: {}", djpeg.display());

    // Create output directories
    let corpus_dir = PathBuf::from("tests/corpus");
    let generated_dir = corpus_dir.join("generated");
    let fuzz_seeds_dir = corpus_dir.join("fuzz_seeds");
    let fixtures_dir = corpus_dir.join("fixtures");

    for dir in [&generated_dir, &fuzz_seeds_dir, &fixtures_dir] {
        if dir.exists() {
            std::fs::remove_dir_all(dir).expect("failed to reset generated corpus directory");
        }
        std::fs::create_dir_all(dir).expect("failed to create output directory");
    }

    // Temp directory for synthetic PPMs
    let tmp_dir = std::env::temp_dir().join("libjpeg_turbo_rs_corpus_ppms");
    std::fs::create_dir_all(&tmp_dir).expect("failed to create temp directory");

    // Prepare source images
    println!("Preparing source PPM images...");
    let sources = prepare_sources(&tmp_dir);
    println!("  {} source images ready", sources.len());

    // Generate JPEG matrix
    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!(
        "Running cjpeg matrix ({} variants × {} sources, {} threads)...",
        all_variants().len(),
        sources.len(),
        parallelism,
    );
    let (generated, failed) = generate_jpegs(&cjpeg, &sources, &generated_dir);
    println!("  generated: {}  failed: {}", generated, failed);

    // Cleanup temp PPMs
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // Copy fuzz seeds
    let fuzz_src = PathBuf::from("fuzz/corpus/fuzz_decompress");
    let fuzz_source_count =
        copy_git_tracked_jpgs(&fuzz_src, &fuzz_seeds_dir).expect("copy tracked fuzz seed corpus");
    let (_, fuzz_parity_count) = retain_matching_files(&fuzz_seeds_dir, &mut |path| {
        strict_djpeg_accepts(&djpeg, path)
    })
    .expect("filter fuzz seeds to strict C parity inputs");
    println!(
        "  fuzz_seeds: {} source files, {} strict C parity inputs",
        fuzz_source_count, fuzz_parity_count
    );

    // Copy fixtures
    let fixtures_src = PathBuf::from("tests/fixtures");
    let fixtures_count = copy_jpgs(&fixtures_src, &fixtures_dir).expect("copy fixture corpus");
    println!("  fixtures: {} files copied", fixtures_count);

    assert_bucket_minimums(
        generated,
        fuzz_source_count,
        fuzz_parity_count,
        fixtures_count,
    )
    .expect("corpus source bucket coverage gate");

    // Summary
    let total = generated + fuzz_parity_count + fixtures_count;
    println!();
    println!("Corpus summary:");
    println!("  generated/  : {}", generated);
    println!("  fuzz_seeds/ : {}", fuzz_parity_count);
    println!("  fixtures/   : {}", fixtures_count);
    println!("  total       : {}", total);
    println!();
    println!(
        "Output: {}",
        corpus_dir.canonicalize().unwrap_or(corpus_dir).display()
    );
}
