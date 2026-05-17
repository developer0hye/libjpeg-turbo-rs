# OSS-Fuzz container for libjpeg-turbo-rs.
#
# Canonical build spec for `google/oss-fuzz/projects/libjpeg-turbo-rs/`.
# This repository owns the source-of-truth; the OSS-Fuzz checkout consumes
# a copy via the upstream PR workflow documented in `oss-fuzz/README.md`.
#
# Build locally (from an OSS-Fuzz checkout):
#   python infra/helper.py build_image libjpeg-turbo-rs
#   python infra/helper.py build_fuzzers --sanitizer address libjpeg-turbo-rs
#   python infra/helper.py run_fuzzer libjpeg-turbo-rs fuzz_decompress

FROM gcr.io/oss-fuzz-base/base-builder-rust

# Minimal build dependencies beyond the Rust base image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/developer0hye/libjpeg-turbo-rs.git $SRC/libjpeg-turbo-rs

WORKDIR $SRC/libjpeg-turbo-rs
COPY build.sh $SRC/
