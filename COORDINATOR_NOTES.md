# Coordinator Notes — Worker B7

Worker: B7 (Multi-thread stress coverage for `TjHandle`, `Encoder`, `Decoder`)
Branch: `worktree-agent-af520495`

This file records workspace-level changes (root `Cargo.toml`, etc.) that the
worker is not permitted to apply directly. The coordinator should apply each
requested change, or explicitly reject with a reason.

---

## WORKSPACE_CARGO_ADDITIONS

Requested edits to `/Cargo.toml` (repository root). All requests are **additive,
dev-only, and cfg-gated** — they cannot affect any release artifact or the
default `cargo test` / `cargo check` / `cargo clippy` code paths.

### 1. `loom` dev-dependency (required for `tests/worker_b7_loom.rs`)

Add a `cfg(loom)`-gated dev-dependency block. `loom` 0.7 is the published stable
permutation checker; activation happens only when the user sets `RUSTFLAGS` to
include `--cfg loom`. Without that flag the dep is neither compiled nor linked.

```toml
# Loom permutation checker for concurrency tests.
# Activated only under `RUSTFLAGS="--cfg loom"`; no effect on normal builds.
# Run loom tests with:
#   RUSTFLAGS="--cfg loom" cargo test --test worker_b7_loom --release
[target.'cfg(loom)'.dev-dependencies]
loom = "0.7"
```

### 2. `unexpected_cfgs` lint config (required so clippy -D warnings still passes)

Rust 1.80+ warns on custom `cfg` names not listed in `check-cfg`. Since
`tests/worker_b7_loom.rs` carries `#![cfg(loom)]`, we need to whitelist the
`loom` cfg or clippy `-D warnings` will fail on any build that touches the
tests target (the pre-commit hook only runs `--lib` so it would not catch this,
but `cargo check --tests` and CI will).

```toml
# `loom` is a known custom cfg used only under `RUSTFLAGS="--cfg loom"`
# for concurrency permutation tests in `tests/worker_b7_loom.rs`.
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(loom)'] }
```

### 3. `rayon` — NOT requested

The B7-2/3/4 mission brief called for rayon-based stress tests. Rayon is
currently a *transitive* dep (via `criterion`) but not a direct dev-dep.

**Decision**: The B7 stress tests use `std::thread::scope` with an explicit
manual work distribution over `N` items instead of `rayon::par_iter`. This
delivers equivalent coverage — N threads concurrently hitting a shared
`&[u8]` source — with zero new workspace deps. See
`tests/worker_b7_concurrency_stress.rs`.

If the coordinator prefers a real `rayon` dep for ergonomics, the following
block can be added:

```toml
[dev-dependencies]
# Already present:
zune-jpeg = "0.5"
# Add:
rayon = "1.10"
```

…and the `thread::scope` bodies in `tests/worker_b7_concurrency_stress.rs`
can be replaced with `(0..N).into_par_iter().for_each(...)`. The assertions
and test intent are unchanged.

---

## FILES ADDED (worktree-local, no workspace edits required)

- `tests/worker_b7_loom.rs` — loom permutation tests (`#![cfg(loom)]`, empty
  unless `RUSTFLAGS="--cfg loom"` is set).
- `tests/worker_b7_concurrency_stress.rs` — rayon-equivalent stress tests via
  `std::thread::scope` (compiles and runs under plain `cargo test`).

Both files are under `tests/` and use only the crate's existing public API.

---

## VERIFICATION (after coordinator applies WORKSPACE_CARGO_ADDITIONS)

```sh
# Default build — loom tests compile empty, stress tests run normally.
cargo test --test worker_b7_concurrency_stress --release

# Loom mode — only worker_b7_loom.rs activates.
RUSTFLAGS="--cfg loom" cargo test --test worker_b7_loom --release

# Lint gate (both must pass):
cargo fmt --all -- --check
cargo clippy --lib -- -D warnings
```
