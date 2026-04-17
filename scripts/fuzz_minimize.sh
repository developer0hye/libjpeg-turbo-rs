#!/usr/bin/env bash
# Minimize and deduplicate every fuzz corpus using `cargo fuzz cmin`.
#
# Why: raw corpus bytes accumulate duplicates and redundant mutations over
# time, bloating commit size and slowing libFuzzer startup. `cargo fuzz cmin`
# merges new inputs into a smaller set that preserves the same edge coverage.
#
# Usage:
#   ./scripts/fuzz_minimize.sh                # minimize all targets
#   ./scripts/fuzz_minimize.sh fuzz_decompress  # one target
#
# Requirements:
#   - rustup toolchain install nightly
#   - cargo install cargo-fuzz
#
# The script is intentionally conservative: it operates on a temp scratch
# directory and only replaces the committed corpus after cmin succeeds, so a
# crash or Ctrl+C never leaves the tree in a broken state.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_BASE="${REPO_ROOT}/fuzz/corpus"

if [[ ! -d "${CORPUS_BASE}" ]]; then
    echo "error: corpus directory not found: ${CORPUS_BASE}" >&2
    exit 1
fi

# Discover fuzz targets. Prefer the authoritative list from `cargo fuzz list`;
# fall back to the on-disk corpus directories if cargo-fuzz isn't installed.
discover_targets() {
    if command -v cargo-fuzz >/dev/null 2>&1 || cargo +nightly fuzz --help >/dev/null 2>&1; then
        (cd "${REPO_ROOT}" && cargo +nightly fuzz list)
    else
        (cd "${CORPUS_BASE}" && ls -1)
    fi
}

if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    mapfile -t TARGETS < <(discover_targets)
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "error: no fuzz targets discovered" >&2
    exit 1
fi

echo "minimizing targets: ${TARGETS[*]}"

for target in "${TARGETS[@]}"; do
    corpus_dir="${CORPUS_BASE}/${target}"
    if [[ ! -d "${corpus_dir}" ]]; then
        echo "skip: no corpus directory for ${target}"
        continue
    fi

    before=$(find "${corpus_dir}" -type f | wc -l | tr -d ' ')
    echo ""
    echo "==> ${target} (before: ${before} files)"

    # Run cmin into a scratch dir so a failed run never leaves the committed
    # corpus partially emptied.
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/fuzz_cmin_${target}.XXXXXX")"
    trap 'rm -rf "${scratch}"' EXIT

    # `cargo fuzz cmin` accepts the target corpus in-place and rewrites it.
    # We use the `-o` form (where available) by copying the corpus into the
    # scratch dir and pointing cmin at it so the original is untouched on
    # failure.
    cp -R "${corpus_dir}/." "${scratch}/"

    if ! (cd "${REPO_ROOT}" && cargo +nightly fuzz cmin "${target}" "${scratch}"); then
        echo "error: cmin failed for ${target}, keeping original corpus" >&2
        rm -rf "${scratch}"
        trap - EXIT
        continue
    fi

    # Replace the committed corpus with the minimized set atomically.
    rm -rf "${corpus_dir}"
    mv "${scratch}" "${corpus_dir}"
    trap - EXIT

    after=$(find "${corpus_dir}" -type f | wc -l | tr -d ' ')
    echo "==> ${target} (after: ${after} files)"
done

echo ""
echo "done. review 'git status' and commit the corpus changes."
