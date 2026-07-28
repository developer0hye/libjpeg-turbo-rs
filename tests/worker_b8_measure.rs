//! Cross-platform peak RSS and wall-clock measurement helpers for B8
//! DoS / pathological-input tests.
//!
//! This file doubles as:
//!   (a) a standalone test binary containing self-tests for the helpers, and
//!   (b) a shared module re-used by other `tests/worker_b8_*.rs` files via
//!       `#[path = "worker_b8_measure.rs"] mod measure;`.
//!
//! Kept outside `tests/helpers/` because that directory is owned by another
//! worker (coordinator guardrail).
//!
//! Peak resident-set size (RSS) is obtained via:
//! - Linux: `/proc/self/status` `VmHWM:` field (kilobytes).
//! - macOS: `task_info` with `MACH_TASK_BASIC_INFO` (`resident_size_max`, bytes).
//! - Other platforms: returns `0` — callers should skip RSS assertions but still
//!   run wall-clock bounds so the test remains useful.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Peak resident-set size and wall-clock duration for a workload.
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    /// Delta peak RSS (bytes) observed over the workload window. `0` when RSS
    /// tracking is unavailable on this platform or when the process high-water
    /// mark did not grow during the workload.
    pub peak_rss_delta_bytes: u64,
    /// Post-workload peak RSS (bytes), absolute. `0` when unsupported.
    pub peak_rss_bytes: u64,
    /// Wall-clock duration.
    pub wall_clock: Duration,
    /// `true` if the running platform supports peak RSS queries.
    pub rss_supported: bool,
}

impl Measurement {
    pub fn peak_rss_delta_mib(&self) -> f64 {
        self.peak_rss_delta_bytes as f64 / (1024.0 * 1024.0)
    }
    pub fn peak_rss_mib(&self) -> f64 {
        self.peak_rss_bytes as f64 / (1024.0 * 1024.0)
    }
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
        peak_rss_bytes: rss_after,
        wall_clock,
        rss_supported: rss_supported(),
    };
    eprintln!(
        "[measure] {}: wall_clock={:.2}ms peak_rss_delta={:.2}MiB peak_rss={:.2}MiB rss_supported={}",
        label,
        measurement.wall_clock_ms(),
        measurement.peak_rss_delta_mib(),
        measurement.peak_rss_mib(),
        measurement.rss_supported,
    );
    (result, measurement)
}

#[cfg(target_os = "linux")]
fn linux_peak_rss_bytes() -> Option<u64> {
    let content: String = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let num_str: &str = rest.split_whitespace().next()?;
            let kb: u64 = num_str.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn macos_peak_rss_bytes() -> Option<u64> {
    use std::mem::MaybeUninit;

    // MACH_TASK_BASIC_INFO = 20, flavor for mach_task_basic_info_data_t.
    const MACH_TASK_BASIC_INFO: i32 = 20;
    // Size of mach_task_basic_info_data_t in natural_t (u32) units.
    //   virtual_size:u64 + resident_size:u64 + resident_size_max:u64
    //   + user_time:(i32,i32) + system_time:(i32,i32) + policy:i32 + suspend_count:i32
    //   = 24 + 8 + 8 + 4 + 4 = 48 bytes = 12 * sizeof(u32)
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
    // SAFETY: info is zero-initialized with the correct size; count is provided
    // by value and read as an out-param by task_info per Darwin ABI.
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
    // SAFETY: task_info returned KERN_SUCCESS.
    let info: MachTaskBasicInfo = unsafe { info.assume_init() };
    Some(info.resident_size_max)
}

// -----------------------------------------------------------------------------
// Self-tests for the helper. This file is its own test binary.
// -----------------------------------------------------------------------------

#[test]
fn peak_rss_returns_reasonable_value_when_supported() {
    let rss: u64 = peak_rss_bytes();
    if rss_supported() {
        assert!(
            rss >= 1024 * 1024,
            "peak RSS should be >=1 MiB on supported platforms, got {}",
            rss
        );
        assert!(
            rss < 16u64 * 1024 * 1024 * 1024,
            "peak RSS implausible: {}",
            rss
        );
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
    assert!(
        m.wall_clock < Duration::from_secs(5),
        "5ms sleep should not take >5s"
    );
}

#[test]
fn measure_large_allocation_is_visible_in_peak_rss_on_supported_platforms() {
    use std::hint::black_box;
    let (_sum, _m) = measure("alloc_16mib", || {
        let n: usize = 16 * 1024 * 1024;
        // black_box() on the Vec prevents release-mode DCE from eliding the
        // allocation entirely — without it the optimizer sees the buffer is
        // dead and never touches physical memory, defeating the RSS probe.
        let mut buf: Vec<u8> = black_box(vec![0u8; n]);
        let mut i: usize = 0;
        let mut acc: u64 = 0;
        while i < n {
            buf[i] = black_box((i & 0xFF) as u8);
            acc = acc.wrapping_add(buf[i] as u64);
            i += 4096;
        }
        black_box(buf);
        acc
    });
    if rss_supported() {
        assert!(
            peak_rss_bytes() >= 16 * 1024 * 1024,
            "after allocating 16 MiB, peak RSS should be >=16 MiB, got {}",
            peak_rss_bytes()
        );
    }
}
