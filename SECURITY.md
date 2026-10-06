# Security Policy

`libjpeg-turbo-rs` decodes untrusted images, so memory-safety and
resource-exhaustion reports are treated as security issues even when the
trigger is "only" a crafted JPEG.

## Supported versions

| Crate | Supported | Gets security fixes |
|---|---|---|
| `libjpeg-turbo-rs` | latest `0.x` minor | latest minor; the previous minor too when the fix is API-compatible with it |
| `libjpeg-turbo-rs-image` | latest `0.x` minor | latest minor |
| `libjpeg-turbo-rs-capi` | latest `0.x` minor | latest minor |
| `libjpeg-turbo-rs-wasm` (npm) | latest release | latest release |

Before 1.0, a minor release may change the API (see
[`docs/STABILITY.md`](docs/STABILITY.md)). A fix that is compatible with the
previous minor ships there as a patch release; a fix that needs an API change
ships only in the next minor, and the advisory says so.

## Reporting a vulnerability

**Do not open a public issue or pull request for a suspected vulnerability.**

Report it privately through GitHub's private vulnerability reporting:
**[Security → Report a vulnerability](https://github.com/developer0hye/libjpeg-turbo-rs/security/advisories/new)**.

If that page says private reporting is not enabled for this repository, open
a public issue titled **"Security contact request"** that contains *no*
details of the problem — no crate, API, file or reproducer — and the
maintainer will open a private advisory and invite you to it.
Include the crate and version, the target (for example `x86_64` with the
default `simd` feature), a reproducer (a JPEG, or the API calls), and what you
observed (crash, sanitizer report, wrong output, unbounded allocation).

What happens next:

1. Acknowledgement within 7 days.
2. Triage on a private advisory: affected versions and targets, severity.
3. A fix on a private branch, with a regression test that fails before it.
4. A release containing the fix, then the published advisory — and a RustSec
   advisory request for issues affecting a published crate.

## Scope

In scope: undefined behaviour reachable from the safe Rust API; out-of-bounds
access, use-after-free or other memory-safety violations through the C ABI
when it is used as `libjpeg`/TurboJPEG documents; process aborts or
unbounded allocation from crafted input that the documented limits
(`DecodeLimits`, `TJPARAM_MAXPIXELS`, `TJPARAM_MAXMEMORY`, `TJPARAM_SCANLIMIT`)
should have bounded.

Out of scope: denial of service bounded only by limits the caller chose not
to set (the defaults intentionally set no memory ceiling and allow frames up
to 2³¹−1 pixels — see
[`docs/STABILITY.md`](docs/STABILITY.md#resource-limits)); bugs that need the
caller to break a documented `unsafe` contract or a C-ABI precondition; output
differences from C libjpeg-turbo with no safety impact (report those as
ordinary issues).

## Known findings and affected versions

[`docs/security/AFFECTED_VERSIONS.md`](docs/security/AFFECTED_VERSIONS.md)
maps every known memory-safety and abort finding to the published versions it
affects and the release that fixes it, and says which warrant an advisory.
Fixed vulnerabilities are also listed in [`CHANGELOG.md`](CHANGELOG.md) under
the release that fixed them, marked **Security**.
