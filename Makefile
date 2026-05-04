.PHONY: check fmt clippy test test-all test-lib test-integration install-hooks install install-capi

# P2-8 install target: stage the libjpeg-turbo-rs-capi cdylib +
# headers + pkg-config + CMake config into ${DESTDIR}${PREFIX} so
# distro packagers can replace upstream libjpeg.so.62 / libturbojpeg.so.0
# without source changes. See scripts/install_capi.sh for the layout.
#
#   make install                  → DESTDIR= PREFIX=/usr/local
#   make install PREFIX=/usr      → unprefixed system install (requires root)
#   make install DESTDIR=/tmp/p   → staged install for packaging
DESTDIR ?=
PREFIX  ?= /usr/local

install: install-capi

install-capi:
	bash scripts/install_capi.sh --build --destdir "$(DESTDIR)" --prefix "$(PREFIX)"


check: fmt clippy test-lib

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

clippy:
	cargo clippy --all-targets -- -D warnings

test-lib:
	cargo test --lib

test-integration:
	cargo test --tests

test-all:
	cargo test

test-cross:
	cargo test cross_encode cross_check --tests

bench:
	cargo bench -- decode_640x480

install-hooks:
	bash .github/hooks/install.sh

clean:
	cargo clean
