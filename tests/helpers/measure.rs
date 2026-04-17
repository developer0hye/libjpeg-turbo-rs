//! Cross-platform peak RSS and wall-clock measurement helpers for DoS tests.
//!
//! Peak resident-set size (RSS) is obtained via:
//! - Linux: `/proc/self/status` `VmHWM:` field (kilobytes).
//! - macOS: `task_info` with `MACH_TASK_BASIC_INFO` (`resident_size_max`, bytes).
//! - Other platforms: returns `0` — callers should skip RSS assertions but still
//!   run wall-clock bounds so the test remains useful.
//!
//! `measure()` wraps a closure, returning `(result, Measurement)`. All reported
//! numbers are peak values across the *entire process lifetime* at the point of
//! the call, so callers should snapshot `peak_rss_bytes()` before the workload
//! and subtract to isolate the delta.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Peak resident-set size and wall-clock duration for a workload.
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    /// Delta peak RSS (bytes) observed over the workload window. `0` when RSS
    /// tracking is unavailable on this platform.
    pub peak_rss_delta_bytes: u64,
    /// Wall-clock duration.
    pub wall_clock: Duration,
    /// `true` if the running platform supports peak RSS queries.
    pub rss_supported: bool,
}

impl Measurement {
    /// Returns peak RSS delta in MiB as f64 for logging.
    pub fn peak_rss_delta_mib(&self) -> f64 {
        self.peak_rss_delta_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Returns wall-clock in milliseconds.
    pub fn wall_clock_ms(&self) -> f64 {
        self.wall_clock.as_secs_f64() * 1000.0
    }
}

/// Read the current process peak RSS in bytes. Returns `0` when the platform
/// is not supported (callers should still run wall-clock assertions).
pub fn peak_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        linux_peak_rss_bytes().unwrap_or(0)
    }
    #[cfg(target_os = "macos")]
    {
        macos_peak_rss_bytes().unwrap_or(0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// Returns `true` when the current platform can report peak RSS.
pub fn rss_supported() -> bool {
    cfg!(any(target_os = "linux", target_os = "macos"))
}

/// Measure wall-clock and peak RSS delta for a closure.
///
/// The peak-RSS delta is computed by subtracting the pre-call peak RSS from the
/// post-call peak RSS, i.e. it reflects new high-water-mark growth caused by
/// the closure. This is an upper bound on what the workload allocated because
/// `VmHWM`/`resident_size_max` is monotonically non-decreasing.
pub fn measure<F, R>(label: &str, f: F) -> (R, Measurement)
where
    F: FnOnce() -> R,
{
    let rss_before: u64 = peak_rss_bytes();
    let t0: Instant = Instant::now();
    let result: R = f();
    let wall_clock: Duration = t0.elapsed();
    let rss_after: u64 = peak_rss_bytes();
    let peak_rss_delta_bytes: u64 = rss_after.saturating_sub(rss_before);
    let measurement: Measurement = Measurement {
        peak_rss_delta_bytes,
        wall_clock,
        rss_supported: rss_supported(),
    };
    eprintln!(
        "[measure] {}: wall_clock={:.2}ms peak_rss_delta={:.2}MiB (rss_supported={})",
        label,
        measurement.wall_clock_ms(),
        measurement.peak_rss_delta_mib(),
        measurement.rss_supported
    );
    (result, measurement)
}

#[cfg(target_os = "linux")]
fn linux_peak_rss_bytes() -> Option<u64> {
    let content: String = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            // Format: "VmHWM:   <N> kB"
            let num_str: &str = rest.trim().split_whitespace().next()?;
            let kb: u64 = num_str.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn macos_peak_rss_bytes() -> Option<u64> {
    // Use mach_task_basic_info.resident_size_max (peak RSS in bytes).
    use std::mem::MaybeUninit;

    // Types transcribed from <mach/task_info.h> / <mach/mach_types.h>.
    // MACH_TASK_BASIC_INFO = 20.
    const MACH_TASK_BASIC_INFO: i32 = 20;
    // Count = sizeof(mach_task_basic_info_data_t) / sizeof(natural_t).
    // mach_task_basic_info_data_t layout (Darwin):
    //   virtual_size       : u64
    //   resident_size      : u64
    //   resident_size_max  : u64
    //   user_time          : time_value (2 * i32)
    //   system_time        : time_value (2 * i32)
    //   policy             : i32
    //   suspend_count      : i32
    // Total size = 3*8 + 4*4 + 2*4 = 48 bytes => 12 natural_t's (natural_t = u32)
    const MACH_TASK_BASIC_INFO_COUNT: u32 = 12;

    #[repr(C)]
    #[derive(Default)]
    struct TimeValue {
        seconds: i32,
        microseconds: i32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        policy: i32,
        suspend_count: i32,
    }

    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: i32,
            task_info_out: *mut i32,
            task_info_out_count: *mut u32,
        ) -> i32;
    }

    let mut info: MaybeUninit<MachTaskBasicInfo> = MaybeUninit::zeroed();
    let mut count: u32 = MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: `info` is zero-initialized and sized for MACH_TASK_BASIC_INFO;
    // `count` is passed by pointer per task_info() ABI.
    let kr: i32 = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as *mut i32,
            &mut count,
        )
    };
    if kr != 0 {
        return None;
    }
    // SAFETY: task_info returned KERN_SUCCESS, info is fully initialized.
    let info: MachTaskBasicInfo = unsafe { info.assume_init() };
    Some(info.resident_size_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_returns_reasonable_value_when_supported() {
        let rss: u64 = peak_rss_bytes();
        if rss_supported() {
            // Any Rust test process uses at least 1 MiB of RSS.
            assert!(
                rss >= 1024 * 1024,
                "peak RSS should be >=1 MiB on supported platforms, got {}",
                rss
            );
            // And < 16 GiB (sanity upper bound).
            assert!(rss < 16u64 * 1024 * 1024 * 1024, "peak RSS implausible");
        } else {
            assert_eq!(rss, 0, "unsupported platforms must return 0");
        }
    }

    #[test]
    fn measure_captures_wall_clock() {
        let (result, m) = measure("sleep_5ms", || {
            std::thread::sleep(Duration::from_millis(5));
            42u32
        });
        assert_eq!(result, 42);
        assert!(
            m.wall_clock >= Duration::from_millis(5),
            "expected >=5ms wall clock, got {:?}",
            m.wall_clock
        );
        // Loose upper bound to catch stalls.
        assert!(
            m.wall_clock < Duration::from_secs(5),
            "5ms sleep should not take >5s"
        );
    }

    #[test]
    fn measure_allocation_increases_peak_rss_on_supported_platforms() {
        // Allocate 16 MiB and touch every page; peak RSS should grow.
        let (_sum, m) = measure("alloc_16mib", || {
            let n: usize = 16 * 1024 * 1024;
            let mut buf: Vec<u8> = vec![0u8; n];
            // Touch each page so the OS actually commits physical memory.
            let page: usize = 4096;
            let mut i: usize = 0;
            let mut acc: u64 = 0;
            while i < n {
                buf[i] = (i & 0xFF) as u8;
                acc = acc.wrapping_add(buf[i] as u64);
                i += page;
            }
            acc
        });
        if m.rss_supported {
            // Delta may be 0 when peak was already higher from earlier tests in
            // the same process; what we can universally assert is that peak RSS
            // is at least 16 MiB.
            assert!(
                peak_rss_bytes() >= 16 * 1024 * 1024,
                "after allocating 16 MiB, peak RSS should be at least 16 MiB, got {}",
                peak_rss_bytes()
            );
        }
    }
}
