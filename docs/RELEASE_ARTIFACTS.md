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
| GitHub release | native bundles, one per target, each with its own `.sha256`, a CycloneDX SBOM (`.cdx.json`, with its own `.sha256`) and two Sigstore bundles (`.provenance.sigstore.json`, `.sbom.sigstore.json`), plus one `SHA256SUMS` covering every archive and SBOM |

The native bundles are new in P4-131. Before them the only way to get a
`libjpeg.so.8` out of this project was to clone the repository, install a Rust
toolchain and run `scripts/install_capi.sh` yourself.

### Native bundles

`libjpeg-turbo-rs-capi-<version>-<target>.tar.gz`, for:

- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`
- `x86_64-pc-windows-msvc`

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
share/doc/libjpeg-turbo-rs-capi/LICENSE-{MIT,APACHE}
include/{jpeglib,jerror,jmorecfg,jconfig,turbojpeg}.h
BUNDLE.txt
```

macOS bundles carry the `.dylib` equivalents (`libjpeg.8.dylib`, …).
`BUNDLE.txt` records the version, target, SONAME, source commit, and the
prefix baked into the `.pc` and CMake files.

The Windows bundle carries upstream's Visual C++ layout instead of a SONAME
chain — the DLL's file name is what a consumer's import table records, so the
name is the identity:

```
bin/jpeg8.dll                   the library, under the name upstream's v8
bin/turbojpeg.dll               (WITH_JPEG8) MSVC build gives it; the same
                                DLL under the TurboJPEG name
lib/jpeg.lib                    import libraries, regenerated from the DLL's
lib/turbojpeg.lib               export table to bind to those names
lib/pkgconfig/, lib/cmake/JPEG/, share/doc/, include/, BUNDLE.txt as above
```

`BUNDLE.txt`'s `soname:` line names the DLL there. The default prefix is
`C:/libjpeg-turbo-rs64`, beside upstream's `c:/libjpeg-turbo64`.

**One staging path.** The bundle is not assembled by the release workflow —
`scripts/package_capi_release.sh` runs `scripts/install_capi.sh` and archives
its output unchanged. That is deliberate: P4-124 requires the downstream
harnesses to test the library we ship, which is only meaningful if there is
one tree to test. `crates/libjpeg-turbo-rs-capi/tests/release_bundle.rs`
enforces it by unpacking a bundle and comparing it entry-by-entry against a
direct install run, and it runs on every pull request on Linux, macOS and
Windows.

## Verifying and installing

```bash
# 1. Verify integrity. SHA256SUMS covers every archive and SBOM in the release.
sha256sum -c SHA256SUMS          # macOS: shasum -a 256 -c SHA256SUMS

# 1b. Verify origin. The attestation is looked up by the archive's digest and
#     checked against Sigstore's transparency log; --source-ref pins it to
#     the tag, so a rehearsal build from a branch cannot pass for a release.
gh attestation verify libjpeg-turbo-rs-capi-<version>-<target>.tar.gz \
    --repo developer0hye/libjpeg-turbo-rs \
    --signer-workflow developer0hye/libjpeg-turbo-rs/.github/workflows/release.yml \
    --source-ref refs/tags/<tag>

# 2. Unpack.
tar -xzf libjpeg-turbo-rs-capi-<version>-<target>.tar.gz

# 3. Install. The default prefix is /usr/local, baked into the .pc and
#    CMake files, so unpacking there needs no further work:
sudo cp -a libjpeg-turbo-rs-capi-<version>-<target>/. /usr/local/
sudo ldconfig                    # Linux
```

`cp -a` rather than `cp -r`: the SONAME chain is symlinks, and copying them as
regular files installs three unrelated copies of the library.

On Windows, from PowerShell (`tar` is bsdtar, shipped since Windows 10 1803):

```powershell
# 1. Verify. `sha256sum -c` from Git for Windows' bash does the same.
(Get-FileHash .\libjpeg-turbo-rs-capi-<version>-x86_64-pc-windows-msvc.tar.gz).Hash
#    — compare with the .sha256 file (case-insensitive).
gh attestation verify libjpeg-turbo-rs-capi-<version>-x86_64-pc-windows-msvc.tar.gz `
    --repo developer0hye/libjpeg-turbo-rs `
    --signer-workflow developer0hye/libjpeg-turbo-rs/.github/workflows/release.yml `
    --source-ref refs/tags/<tag>

# 2. Unpack and install. The .pc and CMake files name C:/libjpeg-turbo-rs64.
tar -xzf libjpeg-turbo-rs-capi-<version>-x86_64-pc-windows-msvc.tar.gz
Copy-Item -Recurse libjpeg-turbo-rs-capi-<version>-x86_64-pc-windows-msvc\* C:\libjpeg-turbo-rs64\
```

Link against `lib\jpeg.lib` (or `lib\turbojpeg.lib`) and put `bin\` on the
`PATH` of whatever loads the DLL, or copy `jpeg8.dll` beside the executable.
The DLL links the dynamic Visual C++ runtime, as upstream's does, so the host
needs the Visual C++ redistributable. MSVC only: `-ljpeg` in the `.pc` files
is `jpeg.lib` under `pkg-config --msvc-syntax`, and no bundle ships the
MinGW names (`libjpeg-8.dll`, `libjpeg.dll.a`).

To install somewhere else, either rewrite the absolute paths in
`lib/pkgconfig/*.pc` and `lib/cmake/JPEG/JPEGConfig.cmake`, or let pkg-config
do it:

```bash
PKG_CONFIG_PATH=<prefix>/lib/pkgconfig pkg-config --define-prefix --cflags --libs libjpeg
```

Building a bundle yourself runs the same script the release does, which adds
only `--target <triple>` and `--sbom`:

```bash
scripts/package_capi_release.sh --outdir dist --build
```

## What is not shipped, and why

### Windows — MSVC only

The Windows bundle is the layout above and nothing more: no MinGW variant, no
installer, no `.pdb`. The import libraries are regenerated because cargo's
own binds every consumer to `libjpeg_turbo_rs_capi.dll`, which the bundle
does not contain; the DLL's bytes are cargo's, unchanged, so its export
directory still records that original name — the loader never reads it, but
a dependency viewer will show it. The scripts refuse any Windows target that
is not `x86_64-*-windows-msvc`, since the `.pc` convention (`-ljpeg`) names a
different file under each toolchain and the bundle can honour only one.

### Signing and SBOM — attested

A checksum published beside the file it covers, on the same host, proves the
download arrived intact and nothing about where it came from. So every bundle
is also **attested**: the `native-artifacts` job that built it signs two
statements about it through [Sigstore](https://www.sigstore.dev/), using the
job's own OIDC identity rather than a key anyone holds, and GitHub stores them
under this repository:

- **Build provenance** (`actions/attest-build-provenance`, SLSA v1): which
  workflow, at which commit, on which runner produced the archive with this
  digest.
- **SBOM** (`actions/attest`, predicate `https://cyclonedx.org/bom`): the
  CycloneDX document `scripts/package_capi_release.sh --sbom` wrote for the
  bundle's target with `cargo-cyclonedx` — the capi crate, the root crate it
  compiles against, and their dependencies, resolved for that target rather
  than for the packaging host.

The attestations are made in the job that produced the bytes, not in
`github-release`, because provenance generated anywhere else would attest a
download rather than a build. They are made unconditionally, so a
`workflow_dispatch` rehearsal exercises the signing path and a tag is never
the first time it runs; the rehearsal that proved it is cited in
[`last_mile/phase4.md` § P4-131](last_mile/phase4.md#p4-131-no-native-binary-distribution--releases-ship-cratesio-and-npm-only--partial-unix-and-windows-bundles-ship-gated-and-attested-the-debrpm-decision-remains).

Verify with the GitHub CLI. The lookup is by the archive's digest, so the file
name does not matter and a renamed download still verifies:

```bash
# Provenance (the default predicate). --signer-workflow pins the workflow
# that signed; --source-ref pins the tag, because a dispatch rehearsal from
# a branch also produces valid attestations and they name the branch.
gh attestation verify <bundle>.tar.gz --repo developer0hye/libjpeg-turbo-rs \
    --signer-workflow developer0hye/libjpeg-turbo-rs/.github/workflows/release.yml \
    --source-ref refs/tags/<tag>

# The SBOM attestation, and the SBOM it signed, without trusting the
# attached .cdx.json copy.
gh attestation verify <bundle>.tar.gz --repo developer0hye/libjpeg-turbo-rs \
    --predicate-type https://cyclonedx.org/bom \
    --format json --jq '.[0].verificationResult.statement.predicate' > sbom.cdx.json

# Offline, from the attached Sigstore bundle. The trusted root can be fetched
# once (`gh attestation trusted-root > trusted_root.jsonl`) and reused.
gh attestation verify <bundle>.tar.gz --repo developer0hye/libjpeg-turbo-rs \
    --bundle <bundle>.tar.gz.provenance.sigstore.json \
    --custom-trusted-root trusted_root.jsonl
```

What a verified attestation proves: the archive with this digest was built by
`release.yml` in this repository at the recorded commit, on a GitHub-hosted
runner, and the SBOM is what that build said its dependency graph was. What it
does not prove: that the source at that commit is trustworthy, or that the
library fits your consumer — the tiers in [`LAST_MILE.md`](LAST_MILE.md)
answer that. The attached `.cdx.json` is a convenience copy; the signed
document is the one inside the attestation, which is why the second command
above extracts it from there.

Upstream libjpeg-turbo signs its source tarballs with a maintainer's GPG key.
This is keyless signing bound to a workflow identity instead; a downloader
who needs a maintainer-held key rather than GitHub's OIDC issuer as the root
of trust does not get that here.

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
2. `native-artifacts` — the bundles, one job per target (four Unix, one
   Windows under bash), each with its SBOM, attested for provenance and SBOM
   in the same job.
3. `publish`, `publish-capi`, `publish-image` — crates.io.
4. `publish-wasm` — npm.
5. `github-release` — creates the release from the CHANGELOG notes and
   attaches the bundles, the SBOMs, their `.sha256` files, the Sigstore
   bundles and a merged `SHA256SUMS`.

Both validations come before the first irreversible step: a registry upload
cannot be withdrawn, so a bundle that fails to build fails ahead of it. The
release itself appears last, so a failed upload — registry or bundle — can never
leave a public release whose downloads are missing.

`workflow_dispatch` runs step 2 alone: every publish job additionally requires
`github.event_name == 'push'`, so a dispatch builds the bundles, uploads them as
workflow artifacts, and publishes nothing — from any ref, including a tag. That
is how to exercise the packaging matrix, including the cross-built
`x86_64-apple-darwin` leg, before a tag makes the output public. The one thing
a dispatch does leave behind is the attestations for its rehearsal bundles,
stored like a tag's and naming the dispatched ref as their source — which is
why the verification commands above pin `--source-ref`.
