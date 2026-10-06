//! What the report records about the machine, toolchain and build.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Run a shell command and return its combined output; a missing tool is
/// reported in the text rather than failing the run (the report says what
/// could not be sampled instead of omitting the row).
pub fn shell(command: &str) -> String {
    match Command::new("sh").arg("-c").arg(command).output() {
        Ok(output) => {
            let mut text: String = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
            if !stderr.trim().is_empty() {
                text.push_str(&stderr);
            }
            if !output.status.success() {
                text.push_str(&format!("[exit status: {}]\n", output.status));
            }
            text.trim_end().to_string()
        }
        Err(error) => format!("[could not run `{command}`: {error}]"),
    }
}

/// CPU and memory load immediately before the measurement. A number without
/// its load context cannot be trusted later.
pub fn load_sample() -> String {
    if cfg!(target_os = "macos") {
        format!(
            "$ top -l 2 -o cpu -n 10 -stats pid,cpu,rsize,command\n{}\n\n$ memory_pressure | tail -1\n{}\n\n$ sysctl vm.swapusage\n{}",
            shell("top -l 2 -o cpu -n 10 -stats pid,cpu,rsize,command"),
            shell("memory_pressure | tail -1"),
            shell("sysctl vm.swapusage"),
        )
    } else if cfg!(target_os = "linux") {
        // top's first frame reports CPU since boot, not now; the second frame,
        // one second later, is the current load.
        format!(
            "$ top -bn2 -d1 | (second frame) | head -20\n{}\n\n$ vmstat 1 3\n{}\n\n$ free -m\n{}",
            shell("top -bn2 -d1 | awk '/^top -/{frame++} frame==2' | head -20"),
            shell("vmstat 1 3"),
            shell("free -m"),
        )
    } else {
        "load sampling is implemented for macOS and Linux only".to_string()
    }
}

pub fn cpu_model() -> String {
    if cfg!(target_os = "macos") {
        shell("sysctl -n machdep.cpu.brand_string")
    } else if cfg!(target_os = "linux") {
        // aarch64 /proc/cpuinfo has no "model name"; lscpu covers both.
        let model: String = shell("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-");
        if model.trim().is_empty() || model.contains("exit status") {
            shell("lscpu | grep -E 'Model name|Vendor ID' | sed 's/  */ /g'")
        } else {
            model.trim().to_string()
        }
    } else {
        "unknown".to_string()
    }
}

pub fn logical_cpus() -> String {
    std::thread::available_parallelism()
        .map(|count| count.get().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

pub fn operating_system() -> String {
    if cfg!(target_os = "macos") {
        format!(
            "macOS {} ({})",
            shell("sw_vers -productVersion"),
            shell("uname -srm")
        )
    } else if cfg!(target_os = "linux") {
        format!(
            "{} ({})",
            shell(". /etc/os-release 2>/dev/null && echo \"$PRETTY_NAME\""),
            shell("uname -srm")
        )
    } else {
        shell("uname -srm")
    }
}

/// Instruction-set extensions the codecs' runtime dispatch can see.
pub fn runtime_cpu_features() -> String {
    #[allow(unused_mut)]
    let mut features: Vec<String> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        let probes: [(&str, bool); 5] = [
            ("sse2", std::arch::is_x86_feature_detected!("sse2")),
            ("sse4.1", std::arch::is_x86_feature_detected!("sse4.1")),
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("bmi2", std::arch::is_x86_feature_detected!("bmi2")),
            ("fma", std::arch::is_x86_feature_detected!("fma")),
        ];
        for (name, present) in probes {
            features.push(format!("{name}={}", if present { "yes" } else { "no" }));
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let neon: bool = std::arch::is_aarch64_feature_detected!("neon");
        features.push(format!("neon={}", if neon { "yes" } else { "no" }));
    }
    format!("{} {}", std::env::consts::ARCH, features.join(" "))
}

/// One `[[package]]` of the consumer's Cargo.lock.
#[derive(Debug, Clone)]
pub struct LockedPackage {
    pub name: String,
    pub version: String,
    pub source: String,
}

/// The resolved dependency set, read from the lock file the build used, so
/// the report states exactly which releases it measured.
pub fn read_lock(path: &Path) -> Vec<LockedPackage> {
    let text: String = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("lock file {} unreadable: {error}", path.display()));
    let mut packages: Vec<LockedPackage> = Vec::new();
    for block in text.split("[[package]]").skip(1) {
        let field = |key: &str| -> Option<String> {
            block.lines().find_map(|line| {
                let rest: &str = line.strip_prefix(key)?.trim_start().strip_prefix('=')?;
                Some(rest.trim().trim_matches('"').to_string())
            })
        };
        packages.push(LockedPackage {
            name: field("name").unwrap_or_default(),
            version: field("version").unwrap_or_default(),
            source: field("source").unwrap_or_else(|| "path".to_string()),
        });
    }
    packages
}

pub fn sha256_hex(path: &Path) -> String {
    let quoted: String = format!("'{}'", path.display().to_string().replace('\'', "'\\''"));
    let output: String = shell(&format!(
        "if command -v sha256sum >/dev/null 2>&1; then sha256sum {quoted}; else shasum -a 256 {quoted}; fi"
    ));
    let digest: &str = output.split_whitespace().next().unwrap_or("");
    assert!(
        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "could not checksum {}: {output}",
        path.display()
    );
    digest.to_string()
}

/// `key=value` lines written by run.sh: build variant, flags, build time,
/// binary and probe sizes, candidate commit. Order is preserved for the
/// report.
pub fn read_build_info(path: Option<&Path>) -> Vec<(String, String)> {
    let Some(path) = path else {
        return vec![(
            "note".to_string(),
            "no --build-info file: run through experiments/downstream/run.sh to record the build"
                .to_string(),
        )];
    };
    let text: String = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("build info {} unreadable: {error}", path.display()));
    text.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

/// The C reference decoder, if one is installed. It is the contract for the
/// libjpeg-turbo-rs rows (pixel-identical to `djpeg` is the project's goal),
/// so where present the report adds candidate-vs-C. Absence is reported, not
/// fatal: hosted runners do not ship djpeg and this harness installs nothing.
///
/// `/usr/local` is deliberately not probed: it is where this repository's own
/// C-ABI shim gets installed, and a `djpeg` linked against the shim would make
/// the "oracle" our own code. Pass `--djpeg` or `DJPEG=` to use one there.
/// The returned path is canonicalised so the report names the real binary,
/// not a symlink in `bin/`.
pub fn find_djpeg(explicit: Option<&Path>) -> Option<PathBuf> {
    let chosen: PathBuf = if let Some(path) = explicit {
        path.to_path_buf()
    } else if let Ok(path) = std::env::var("DJPEG") {
        PathBuf::from(path)
    } else {
        [
            "/opt/homebrew/bin/djpeg",
            "/opt/libjpeg-turbo/bin/djpeg",
            "/usr/bin/djpeg",
        ]
        .into_iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.is_file())?
    };
    Some(std::fs::canonicalize(&chosen).unwrap_or(chosen))
}

/// The dynamic libraries `djpeg` resolves, so the report shows which libjpeg
/// actually produced the C pixels (a `djpeg` binary says nothing about the
/// library it loads).
pub fn djpeg_link_map(path: &Path) -> String {
    let quoted: String = format!("'{}'", path.display().to_string().replace('\'', "'\\''"));
    if cfg!(target_os = "macos") {
        // `@rpath/...` entries resolve through the binary's LC_RPATH list.
        format!(
            "$ otool -L {quoted}\n{}\n$ otool -l {quoted} | grep -A2 LC_RPATH | grep path\n{}",
            shell(&format!("otool -L {quoted}")),
            shell(&format!(
                "otool -l {quoted} | grep -A2 LC_RPATH | grep path || true"
            ))
        )
    } else {
        format!("$ ldd {quoted}\n{}", shell(&format!("ldd {quoted}")))
    }
}
