# Release artifacts

What a tagged release publishes, how to verify it, and what it is still
missing. The compatibility question — *may I replace my system libjpeg with
this?* — is answered in [`ABI_COMPATIBILITY.md`](ABI_COMPATIBILITY.md) and by
the T1–T4 tiers in [`LAST_MILE.md`](LAST_MILE.md). Read those first: this page
is about delivery, not about whether the thing delivered fits.

## What ships

| Channel | Artifact |
| --- | --- |
| crates.io | `libjpeg-turbo-rs`, `libjpeg-turbo-rs-capi`, `libjpeg-turbo-rs-image` |
| npm | `libjpeg-turbo-rs-wasm` |
| GitHub release | native bundles, one per platform, plus `SHA256SUMS` |

The native bundles are new in P4-131. Before them the only way to get a
`libjpeg.so.8` out of this project was to clone the repository, install a Rust
toolchain and run `scripts/install_capi.sh` yourself.

### Native bundles

`libjpeg-turbo-rs-capi-<version>-<target>.tar.gz`, for:

- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`

Each unpacks into a single directory holding the prefix
`scripts/install_capi.sh` stages:

```
lib/libjpeg.so.8.X.Y            the library (relinked with GNU symbol
lib/libjpeg.so.8      → .X.Y     versions on Linux — P4-81)
lib/libjpeg.so        → .so.8
lib/libturbojpeg.so.0.X.Y       the same binary; we export both APIs
lib/libturbojpeg.so.0 → .X.Y
lib/libturbojpeg.so   → .so.0
lib/pkgconfig/libjpeg.pc
lib/pkgconfig/libturbojpeg.pc
lib/cmake/JPEG/JPEGConfig.cmake
include/{jpeglib,jerror,jmorecfg,jconfig,turbojpeg}.h
BUNDLE.txt
```

macOS bundles carry the `.dylib` equivalents (`libjpeg.8.dylib`, …).
`BUNDLE.txt` records the version, target, SONAME, source commit, and the
prefix baked into the `.pc` and CMake files.

**One staging path.** The bundle is not assembled by the release workflow —
`scripts/package_capi_release.sh` runs `scripts/install_capi.sh` and archives
its output unchanged. That is deliberate: P4-124 requires the downstream
harnesses to test the library we ship, which is only meaningful if there is
one tree to test. `crates/libjpeg-turbo-rs-capi/tests/release_bundle.rs`
enforces it by unpacking a bundle and comparing it entry-by-entry against a
direct install run, and it runs on every pull request on Linux and macOS.

## Verifying and installing

```bash
# 1. Verify. SHA256SUMS covers every bundle in the release.
sha256sum -c SHA256SUMS          # macOS: shasum -a 256 -c SHA256SUMS

# 2. Unpack.
tar -xzf libjpeg-turbo-rs-capi-<version>-<target>.tar.gz

# 3. Install. The default prefix is /usr/local, baked into the .pc and
#    CMake files, so unpacking there needs no further work:
sudo cp -a libjpeg-turbo-rs-capi-<version>-<target>/. /usr/local/
sudo ldconfig                    # Linux
```

`cp -a` rather than `cp -r`: the SONAME chain is symlinks, and copying them as
regular files installs three unrelated copies of the library.

To install somewhere else, either rewrite the absolute paths in
`lib/pkgconfig/*.pc` and `lib/cmake/JPEG/JPEGConfig.cmake`, or let pkg-config
do it:

```bash
PKG_CONFIG_PATH=<prefix>/lib/pkgconfig pkg-config --define-prefix --cflags --libs libjpeg
```

Building the bundle yourself is the same one command the release runs:

```bash
scripts/package_capi_release.sh --outdir dist --build
```

## What is not shipped, and why

### Windows — open

No DLL or import library. `install_capi.sh` handles Linux and macOS only, and
the packaging script refuses to run elsewhere rather than emit a bundle whose
shape nothing has verified. A Windows bundle needs its own layout decision
(no SONAME chain, an import library, and a `.pc` convention that differs by
toolchain), so it is a separate piece of work rather than a flag. Tracked
under **P4-131** ([#462](https://github.com/developer0hye/libjpeg-turbo-rs/issues/462)),
which stays PARTIAL until it lands.

### Signing and SBOM — a recorded gap

Bundles are checksummed but **not signed**, and no SBOM is published.

The reason recorded before P4-131 was sequencing — signing is only meaningful
once there is an artifact to sign, and there was not one. That reason is now
spent. What replaces it is narrower: a checksum published beside the file it
covers, on the same host, proves integrity of the download and nothing about
its origin. Closing that means Sigstore build provenance
(`actions/attest-build-provenance`, verified with `gh attestation verify`) or
detached signatures, and either one is only observable on a real tagged run —
there is no way to prove it works from a pull request. Wiring an unverifiable
step into the path that creates releases is how a release breaks at the worst
moment, so it is left for a change that can be dispatched and checked
end-to-end first. `workflow_dispatch` on the release workflow exists for
exactly that.

Until then: **a downloaded bundle's authenticity rests on GitHub's transport
and account security, not on a signature you can check offline.** Upstream
libjpeg-turbo ships signed tarballs; this is a real gap against them, not a
difference of opinion.

### Distro packaging (deb/rpm) — undecided

Still neither in scope nor a recorded non-goal, and it is not resolved here
because it is not a technical question. The tarballs give a distribution
packager everything a `debian/rules` or `%install` needs, so the remaining
question is whether this project wants to *be* the packager for consumers who
are most exposed to the open T3 gaps — a maintainer decision.
[#462](https://github.com/developer0hye/libjpeg-turbo-rs/issues/462) carries
it.

## How a release is produced

`.github/workflows/release.yml`, on a `v*` tag:

1. `changelog-check` — the tag must have a CHANGELOG section.
2. `publish`, `publish-capi`, `publish-image` — crates.io.
3. `publish-wasm` — npm.
4. `native-artifacts` — the bundles, one job per target.
5. `github-release` — creates the release from the CHANGELOG notes and
   attaches the bundles with a merged `SHA256SUMS`.

The release appears last, so a failed upload — registry or bundle — can never
leave a public release whose downloads are missing.

`workflow_dispatch` runs step 4 alone: every publish job is gated on a tag
ref, so a dispatch builds the bundles, uploads them as workflow artifacts, and
publishes nothing. That is how to exercise the packaging matrix, including the
cross-built `x86_64-apple-darwin` leg, before a tag makes the output public.
