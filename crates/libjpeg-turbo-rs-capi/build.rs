// Build script for libjpeg-turbo-rs-capi.
//
// Filled out in later subtasks:
// - A1-13: SONAME (Linux) / install_name (macOS) for drop-in compatibility.
// - A1-14: pkg-config `.pc` file generation.
//
// Kept intentionally minimal for the A1-1 scaffold step.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
