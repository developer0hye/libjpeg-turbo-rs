#!/usr/bin/env bash
# OSS-Fuzz build script for libjpeg-turbo-rs.
#
# Invoked by OSS-Fuzz's base-builder-rust image with the standard env vars:
#   $SRC        — source root (populated by Dockerfile)
#   $OUT        — directory where compiled fuzzers must be placed
#   $SANITIZER  — address|memory|undefined|coverage
#
# cargo-fuzz emits one binary per target under fuzz/target/<triple>/release.
# We copy each binary (and its seed corpus) into $OUT with the canonical name
# expected by OSS-Fuzz (<target_name> and <target_name>_seed_corpus.zip).

set -euo pipefail

cd "${SRC}/libjpeg-turbo-rs"

# Regenerate the corpus so every fuzz target starts from meaningful seeds.
cargo test --test generate_fuzz_seeds

# cargo-fuzz is preinstalled in base-builder-rust, but enforce a known version
# for reproducibility.
cargo install --locked cargo-fuzz --version "0.12.0" || true

TARGETS=(
    fuzz_decompress
    fuzz_decompress_lenient
    fuzz_roundtrip
    fuzz_read_coefficients
    fuzz_transform
    fuzz_progressive_decoder
    fuzz_encode_roundtrip
)

# Forward OSS-Fuzz's libFuzzer-compatible flags to cargo fuzz.
FUZZ_FLAGS=(-O --release)

for target in "${TARGETS[@]}"; do
    echo "==> building ${target}"
    cargo fuzz build "${target}" "${FUZZ_FLAGS[@]}"

    bin_path="$(find fuzz/target -type f -name "${target}" -perm -u+x | head -n1)"
    if [[ -z "${bin_path}" ]]; then
        echo "error: could not locate built binary for ${target}" >&2
        exit 1
    fi
    cp "${bin_path}" "${OUT}/${target}"

    # Package the seed corpus into a zip for OSS-Fuzz's seed-corpus loader.
    corpus_dir="fuzz/corpus/${target}"
    if [[ -d "${corpus_dir}" ]]; then
        (cd "${corpus_dir}" && zip -qr "${OUT}/${target}_seed_corpus.zip" .)
    fi
done

echo "oss-fuzz build finished; emitted ${#TARGETS[@]} fuzzers to ${OUT}"
