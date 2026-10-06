#!/usr/bin/env bash
# API compatibility gate for a release (#638, docs/STABILITY.md).
#
# Compares each crates.io crate's public API at HEAD with the same crate at
# <baseline-ref> (normally the previous release tag) and fails when the
# version change does not allow what changed: a 0.x patch bump must not
# break the API, a 0.x minor bump may.
#
#   scripts/semver_check_release.sh <baseline-ref> [crate ...]
#
# Why rustdoc files rather than `cargo semver-checks check-release` on its
# own: the tool's default mode builds both sides in a scratch workspace that
# resolves dependencies fresh from crates.io, ignoring Cargo.lock, and there
# is no `--locked` for it. That runs whatever build scripts and proc macros
# the registry serves at that minute — the hole the 2026-08-20 crates.io
# supply-chain attack came through. Here both sides are documented with
# `cargo rustdoc --locked` against their own committed Cargo.lock, and the
# tool only compares the two JSON files.
#
# Requires: cargo-semver-checks (CI installs a pinned version with --locked),
# python3 >= 3.11 (tomllib), a git checkout that contains <baseline-ref>.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <baseline-ref> [crate ...]" >&2
    exit 2
fi
baseline_ref="$1"
shift
if [ "$#" -gt 0 ]; then
    crates=("$@")
else
    crates=(libjpeg-turbo-rs libjpeg-turbo-rs-capi libjpeg-turbo-rs-image)
fi

repo_root="$(git rev-parse --show-toplevel)"
scratch="$(mktemp -d)"
baseline_tree="$scratch/baseline"
cleanup() {
    git -C "$repo_root" worktree remove --force "$baseline_tree" >/dev/null 2>&1 || true
    rm -rf "$scratch"
}
trap cleanup EXIT
git -C "$repo_root" worktree add --quiet --detach "$baseline_tree" "$baseline_ref"

manifest_of() {
    # <tree> <crate> -> path of that crate's Cargo.toml, or nothing.
    if [ "$2" = "libjpeg-turbo-rs" ]; then
        echo "$1/Cargo.toml"
    elif [ -f "$1/crates/$2/Cargo.toml" ]; then
        echo "$1/crates/$2/Cargo.toml"
    fi
}

version_of() {
    python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["package"]["version"])' "$1"
}

# <previous> <current> -> the cargo-semver-checks release type that permits
# exactly what Cargo's caret rules let a dependent receive. Rustdoc input
# carries no version, and the tool's own types are 1.x-shaped: for 0.y.z,
# a minor bump is Cargo's "breaking" step (`major`) and a patch bump its
# "compatible" step (`minor`).
release_type() {
    python3 - "$1" "$2" <<'PY'
import sys
old = [int(x) for x in sys.argv[1].split("-")[0].split(".")]
new = [int(x) for x in sys.argv[2].split("-")[0].split(".")]
if new <= old:
    sys.exit(f"version did not increase: {sys.argv[1]} -> {sys.argv[2]}")
if old[0] == 0:
    print("major" if (new[0], new[1]) != (old[0], old[1]) else "minor")
else:
    print("major" if new[0] != old[0] else "minor" if new[1] != old[1] else "patch")
PY
}

rustdoc_json() {
    # <tree> <crate> <target-dir> -> path of the crate's rustdoc JSON
    (cd "$1" && RUSTC_BOOTSTRAP=1 cargo rustdoc --locked --quiet -p "$2" --lib \
        --target-dir "$3" -- -Z unstable-options --output-format json)
    echo "$3/doc/${2//-/_}.json"
}

status=0
for crate in "${crates[@]}"; do
    current_manifest="$(manifest_of "$repo_root" "$crate")"
    baseline_manifest="$(manifest_of "$baseline_tree" "$crate")"
    if [ -z "$current_manifest" ]; then
        echo "error: no crate $crate at HEAD" >&2
        exit 2
    fi
    if [ -z "$baseline_manifest" ]; then
        echo "$crate: not present at $baseline_ref, nothing to compare"
        continue
    fi
    current_version="$(version_of "$current_manifest")"
    baseline_version="$(version_of "$baseline_manifest")"
    if [ "$current_version" = "$baseline_version" ]; then
        # The publish step skips an already-published version, so nothing of
        # this crate ships and there is no version number to check.
        echo "$crate: still $current_version, not released by this tag — skipped"
        continue
    fi
    kind="$(release_type "$baseline_version" "$current_version")"
    echo "$crate: $baseline_version -> $current_version (checked as a '$kind' change)"
    baseline_json="$(rustdoc_json "$baseline_tree" "$crate" "$scratch/target-baseline")"
    current_json="$(rustdoc_json "$repo_root" "$crate" "$scratch/target-current")"
    if ! cargo semver-checks check-release \
        --baseline-rustdoc "$baseline_json" \
        --current-rustdoc "$current_json" \
        --release-type "$kind"; then
        echo "::error::$crate $baseline_version -> $current_version breaks the API more than the version allows"
        status=1
    fi
done
exit "$status"
