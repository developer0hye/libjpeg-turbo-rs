# Documentation Map

One line per document. Status and readiness are stated in exactly one place,
[`LAST_MILE.md`](LAST_MILE.md); the others link to it rather than restate it.

## Adopting

| Document | Answers |
| --- | --- |
| [`../README.md`](../README.md) | What the project is, quick start, measured performance |
| [`ADOPTION_GUIDE.md`](ADOPTION_GUIDE.md) | Which integration path to use, what to pin, how to evaluate and roll out |
| [`STABILITY.md`](STABILITY.md) | SemVer, MSRV, features, errors, threads and resource-limit policy |
| [`../SECURITY.md`](../SECURITY.md) | Supported versions and how to report a vulnerability |
| [`security/AFFECTED_VERSIONS.md`](security/AFFECTED_VERSIONS.md) | Which published versions carry which known safety or abort finding |
| [`../CHANGELOG.md`](../CHANGELOG.md) | What changed per release, and what is unreleased |
| [`../crates/libjpeg-turbo-rs-image/README.md`](../crates/libjpeg-turbo-rs-image/README.md) | The `image` adapter's contract |
| [`../crates/libjpeg-turbo-rs-wasm/README.md`](../crates/libjpeg-turbo-rs-wasm/README.md) | WebAssembly build and distribution |
| [`../crates/libjpeg-turbo-rs-capi/README.md`](../crates/libjpeg-turbo-rs-capi/README.md) | Building and installing the C ABI |

## C ABI

| Document | Answers |
| --- | --- |
| [`C_API_REFERENCE.md`](C_API_REFERENCE.md) | Every C function and its Rust status |
| [`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md) | ABI, SONAME, threading and legacy-alias policy |
| [`RELEASE_ARTIFACTS.md`](RELEASE_ARTIFACTS.md) | Native bundles: contents, verification, installation |

## Status and evidence

| Document | Answers |
| --- | --- |
| [`LAST_MILE.md`](LAST_MILE.md) | Readiness tiers T1–T4, the live gate, every open gap |
| [`FEATURE_PARITY.md`](FEATURE_PARITY.md) | Which libjpeg-turbo features are implemented |
| [`TEST_PARITY.md`](TEST_PARITY.md), [`CORPUS_TEST_REPORT.md`](CORPUS_TEST_REPORT.md) | How behaviour is cross-validated against C |
| [`UNSAFE_INVENTORY.md`](UNSAFE_INVENTORY.md), [`UNSAFE_INVENTORY_CAPI.md`](UNSAFE_INVENTORY_CAPI.md) | Every `unsafe` item, its invariant and its test |
| [`oracle_versions.tsv`](oracle_versions.tsv) | The C oracle versions the gates pin |
| [`ENCODING_PERFORMANCE.md`](ENCODING_PERFORMANCE.md), [`../experiments/`](../experiments/README.md) | Dated benchmark evidence |

## Maintaining

| Document | Answers |
| --- | --- |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Workflow and pull-request rules |
| [`RELEASE.md`](RELEASE.md) | How a release is cut and verified |
| [`last_mile/`](last_mile/) | Per-phase detail behind each LAST_MILE item |

The pre-1.0 public-surface review (`PUBLIC_API_REVIEW.md`, P4-222) is in PR
[#649](https://github.com/developer0hye/libjpeg-turbo-rs/pull/649) and will be
added here when it merges.
