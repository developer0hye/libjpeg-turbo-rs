.PHONY: check fmt clippy test test-all test-lib test-integration install-hooks

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
