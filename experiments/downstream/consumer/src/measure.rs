//! Timing statistics, pixel comparison, PSNR, and JPEG header inspection.

use std::time::Duration;

/// Summary of one row's timed iterations.
#[derive(Debug, Clone, Copy)]
pub struct TimingSummary {
    pub iterations: usize,
    pub median_ms: f64,
    pub p10_ms: f64,
    pub p90_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

/// Nearest-rank percentile over sorted samples. With the smoke run's two
/// samples p10 is the minimum and p90 the maximum, which is what the report
/// should say for so few samples rather than an interpolated fiction.
fn nearest_rank(sorted: &[f64], percentile: f64) -> f64 {
    let rank: usize = ((percentile / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

pub fn summarize(samples: &[Duration]) -> TimingSummary {
    assert!(!samples.is_empty(), "a timed row needs at least one sample");
    let mut milliseconds: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    milliseconds.sort_by(|a, b| a.total_cmp(b));
    let count: usize = milliseconds.len();
    let median_ms: f64 = if count % 2 == 1 {
        milliseconds[count / 2]
    } else {
        (milliseconds[count / 2 - 1] + milliseconds[count / 2]) / 2.0
    };
    TimingSummary {
        iterations: count,
        median_ms,
        p10_ms: nearest_rank(&milliseconds, 10.0),
        p90_ms: nearest_rank(&milliseconds, 90.0),
        min_ms: milliseconds[0],
        max_ms: milliseconds[count - 1],
    }
}

/// Per-sample difference between two equally sized pixel buffers.
#[derive(Debug, Clone, Copy)]
pub struct PixelDiff {
    pub max_abs: u8,
    pub mean_abs: f64,
    /// Samples (not pixels) that differ at all.
    pub differing_samples: usize,
}

pub fn pixel_diff(left: &[u8], right: &[u8]) -> PixelDiff {
    assert_eq!(
        left.len(),
        right.len(),
        "pixel_diff needs equally sized buffers; callers check dimensions first"
    );
    let mut max_abs: u8 = 0;
    let mut total: u64 = 0;
    let mut differing_samples: usize = 0;
    for (a, b) in left.iter().zip(right) {
        let difference: u8 = a.abs_diff(*b);
        if difference != 0 {
            differing_samples += 1;
            total += difference as u64;
            max_abs = max_abs.max(difference);
        }
    }
    PixelDiff {
        max_abs,
        mean_abs: if left.is_empty() {
            0.0
        } else {
            total as f64 / left.len() as f64
        },
        differing_samples,
    }
}

/// PSNR in dB over all samples; `f64::INFINITY` for identical buffers.
pub fn psnr(reference: &[u8], test: &[u8]) -> f64 {
    assert_eq!(reference.len(), test.len(), "PSNR needs equal sizes");
    let squared_error: u64 = reference
        .iter()
        .zip(test)
        .map(|(a, b)| {
            let d: u64 = a.abs_diff(*b) as u64;
            d * d
        })
        .sum();
    if squared_error == 0 {
        return f64::INFINITY;
    }
    let mse: f64 = squared_error as f64 / reference.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Frame facts read from a JPEG's SOF marker. Encoders do not all honour a
/// requested subsampling (image's encoder picks its own), so the report states
/// what each output actually contains instead of what was asked for.
#[derive(Debug, Clone)]
pub struct FrameFacts {
    pub width: u16,
    pub height: u16,
    pub process: &'static str,
    pub subsampling: String,
}

pub fn inspect_frame(jpeg: &[u8]) -> Option<FrameFacts> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return None;
    }
    let mut position: usize = 2;
    while position + 4 <= jpeg.len() {
        if jpeg[position] != 0xFF {
            return None;
        }
        let marker: u8 = jpeg[position + 1];
        if marker == 0xFF {
            position += 1; // fill byte
            continue;
        }
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            position += 2;
            continue;
        }
        let length: usize = u16::from_be_bytes([jpeg[position + 2], jpeg[position + 3]]) as usize;
        let segment_start: usize = position + 4;
        let is_sof: bool = (0xC0..=0xCF).contains(&marker) && ![0xC4, 0xC8, 0xCC].contains(&marker);
        if is_sof {
            let segment: &[u8] = jpeg.get(segment_start..position + 2 + length)?;
            let height: u16 = u16::from_be_bytes([segment[1], segment[2]]);
            let width: u16 = u16::from_be_bytes([segment[3], segment[4]]);
            let component_count: usize = segment[5] as usize;
            let mut factors: Vec<(u8, u8)> = Vec::with_capacity(component_count);
            for component in 0..component_count {
                let sampling: u8 = *segment.get(6 + component * 3 + 1)?;
                factors.push((sampling >> 4, sampling & 0x0F));
            }
            let process: &'static str = match marker {
                0xC0 => "baseline",
                0xC1 => "extended",
                0xC2 => "progressive",
                0xC3 => "lossless",
                _ => "other",
            };
            return Some(FrameFacts {
                width,
                height,
                process,
                subsampling: describe_sampling(&factors),
            });
        }
        if marker == 0xDA {
            return None; // scan before any frame header
        }
        position = position + 2 + length;
    }
    None
}

fn describe_sampling(factors: &[(u8, u8)]) -> String {
    match factors {
        [_] => "gray".to_string(),
        [(lh, lv), (h1, v1), (h2, v2)] if (h1, v1) == (&1, &1) && (h2, v2) == (&1, &1) => {
            match (lh, lv) {
                (1, 1) => "4:4:4".to_string(),
                (2, 1) => "4:2:2".to_string(),
                (2, 2) => "4:2:0".to_string(),
                (1, 2) => "4:4:0".to_string(),
                (4, 1) => "4:1:1".to_string(),
                _ => format!("{lh}x{lv},1x1,1x1"),
            }
        }
        _ => factors
            .iter()
            .map(|(h, v)| format!("{h}x{v}"))
            .collect::<Vec<String>>()
            .join(","),
    }
}

/// Parse a binary PPM (P6) or PGM (P5) with maxval 255, as written by djpeg.
pub fn parse_pnm(bytes: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let channels: usize = match bytes.get(0..2)? {
        b"P6" => 3,
        b"P5" => 1,
        _ => return None,
    };
    let mut fields: Vec<usize> = Vec::with_capacity(3);
    let mut position: usize = 2;
    while fields.len() < 3 {
        while bytes.get(position)?.is_ascii_whitespace() {
            position += 1;
        }
        if bytes[position] == b'#' {
            while *bytes.get(position)? != b'\n' {
                position += 1;
            }
            continue;
        }
        let start: usize = position;
        while bytes.get(position)?.is_ascii_digit() {
            position += 1;
        }
        fields.push(
            std::str::from_utf8(&bytes[start..position])
                .ok()?
                .parse()
                .ok()?,
        );
    }
    position += 1; // the single whitespace byte after maxval
    let (width, height, maxval) = (fields[0], fields[1], fields[2]);
    if maxval != 255 {
        return None;
    }
    let data: Vec<u8> = bytes
        .get(position..position + width * height * channels)?
        .to_vec();
    Some((width, height, channels, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles_on_ten_samples() {
        let samples: Vec<Duration> = (1..=10).map(Duration::from_millis).collect();
        let summary: TimingSummary = summarize(&samples);
        assert_eq!(summary.median_ms, 5.5);
        assert_eq!(summary.p10_ms, 1.0);
        assert_eq!(summary.p90_ms, 9.0);
        assert_eq!(summary.min_ms, 1.0);
        assert_eq!(summary.max_ms, 10.0);
    }

    #[test]
    fn psnr_of_identical_buffers_is_infinite() {
        assert!(psnr(&[1, 2, 3], &[1, 2, 3]).is_infinite());
        // One sample off by 1 in three: MSE = 1/3.
        let expected: f64 = 10.0 * (255.0f64 * 255.0 * 3.0).log10();
        assert!((psnr(&[1, 2, 3], &[1, 2, 4]) - expected).abs() < 1e-9);
    }

    #[test]
    fn pnm_header_with_comment() {
        let mut bytes: Vec<u8> = b"P6\n# c\n2 1\n255\n".to_vec();
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let (width, height, channels, data) = parse_pnm(&bytes).expect("valid PPM");
        assert_eq!((width, height, channels), (2, 1, 3));
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6]);
    }
}
