//! P4-135 criterion 2 (#474): the SIMD kernel modules must stay unreachable
//! from outside the crate.
//!
//! `pub mod simd` used to let any downstream crate call a `target_feature`
//! kernel by path — the proof-of-concept this issue opens with. The arch
//! modules are now `pub(crate)`, and building an external crate against the
//! library fails with `error[E0603]: module 'x86_64' is private`.
//!
//! # Why this asserts the source rather than using `trybuild`
//!
//! The criterion names `trybuild`. It is not used here, deliberately:
//!
//! * it would add a dev-dependency for one assertion, and
//! * its stderr snapshots pin exact rustc diagnostics, which drift between
//!   compiler releases — this repo builds on MSRV 1.87 *and* stable, so a
//!   snapshot correct on one would fail the other, and the usual fix is to
//!   loosen the snapshot until it stops proving anything.
//!
//! What actually regresses is someone widening `pub(crate)` back to `pub`.
//! That is a source-level edit, so this checks the source — the same guard
//! shape `capi_symbol_versions.rs` and `capi_classic_state_constants.rs`
//! already use, and it runs in milliseconds.
//!
//! This test lives in `tests/` on purpose: an in-crate test could not observe
//! the distinction, since `pub(crate)` is visible to it either way.
//!
//! # Why this is skipped on wasm
//!
//! Reading the source tree requires a filesystem. Under `wasm32-wasip1` the
//! runner (wasmtime) only exposes directories it was explicitly given, and CI
//! preopens none — so `read_to_string` fails and, with `panic = "abort"`,
//! takes the whole test binary down rather than reporting a failure.
//!
//! The guard is *inapplicable* there, not merely unrunnable: what it checks is
//! one line of committed source, identical no matter which target is being
//! built. Every non-wasm job in the matrix runs it, so a `pub mod` regression
//! still cannot reach `main`.

#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;

/// Arch backends whose kernels must not be externally callable. `scalar` is
/// included: it is a fallback, not API.
const PRIVATE_BACKENDS: &[&str] = &["scalar", "aarch64", "x86_64", "wasm32"];

fn simd_mod_source() -> String {
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("simd")
        .join("mod.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn simd_kernel_modules_are_not_public() {
    let source: String = simd_mod_source();

    let mut public: Vec<&str> = Vec::new();
    for backend in PRIVATE_BACKENDS {
        let public_decl: String = format!("pub mod {backend};");
        let private_decl: String = format!("pub(crate) mod {backend};");
        // `pub mod x;` must not appear; `pub(crate) mod x;` must.
        if source.contains(&public_decl) {
            public.push(backend);
        }
        assert!(
            source.contains(&private_decl),
            "`{backend}` is not declared `pub(crate)` in src/simd/mod.rs — if it \
             was renamed or removed, update PRIVATE_BACKENDS rather than deleting \
             this assertion"
        );
    }

    assert!(
        public.is_empty(),
        "these SIMD backends are `pub`, so a downstream crate can call their \
         `target_feature` kernels by path with no `unsafe` at the call site \
         (P4-135): {public:?}"
    );
}

/// The crate root must not re-export the kernels around the module privacy.
#[test]
fn simd_kernels_are_not_re_exported_from_the_crate_root() {
    let lib: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("lib.rs");
    let text: String = std::fs::read_to_string(&lib).expect("read src/lib.rs");

    for backend in PRIVATE_BACKENDS {
        let leak: String = format!("pub use crate::simd::{backend}");
        let leak2: String = format!("pub use simd::{backend}");
        assert!(
            !text.contains(&leak) && !text.contains(&leak2),
            "src/lib.rs re-exports `simd::{backend}`, which reopens the path \
             module privacy just closed"
        );
    }
}
