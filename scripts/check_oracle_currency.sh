#!/usr/bin/env bash
#
# P4-130 criterion 4: notice when upstream libjpeg-turbo moves past the release
# our differential gates prove parity with.
#
# 3.2.0 shipped 2026-06-30 and the oracle pins still said 3.1.4.1 on
# 2026-08-09. Nothing was broken by that — 3.1.4.1 is a real supported release —
# but nothing *reported* it either, so "our gates prove parity with current
# upstream" quietly stopped being true and stayed that way for two months. The
# policy is therefore mechanical rather than a promise to remember: this script
# compares the `tool-current` row of docs/oracle_versions.tsv against upstream's
# latest stable release and fails when they differ. It runs weekly from
# .github/workflows/upstream-currency.yml, so a new upstream release turns a
# scheduled job red within a week of shipping.
#
# Failing is the notification. What to do about it is a judgement call — a new
# minor may want its own leg, a patch release may just move `tool-current` —
# and the answer belongs in P4-130 / a follow-up gap, not in this script.
#
# Usage: scripts/check_oracle_currency.sh [path/to/oracle_versions.tsv]
# Env:   GITHUB_TOKEN  optional; raises the GitHub API rate limit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$REPO_ROOT/docs/oracle_versions.tsv}"
UPSTREAM_API="https://api.github.com/repos/libjpeg-turbo/libjpeg-turbo/releases/latest"

if [ ! -f "$MANIFEST" ]; then
    echo "error: oracle manifest not found: $MANIFEST" >&2
    exit 1
fi

# TAB-separated `role<TAB>version<TAB>...`; comments start with '#'.
declared="$(awk -F'\t' '$1 == "tool-current" { print $2 }' "$MANIFEST")"
declared_count="$(printf '%s\n' "$declared" | grep -c . || true)"
if [ "$declared_count" -ne 1 ]; then
    echo "error: $MANIFEST must declare exactly one tool-current row; found $declared_count" >&2
    exit 1
fi

curl_args=(-fsSL -H "Accept: application/vnd.github+json")
if [ -n "${GITHUB_TOKEN:-}" ]; then
    curl_args+=(-H "Authorization: Bearer $GITHUB_TOKEN")
fi

# `/releases/latest` excludes prereleases, so a 3.3 beta does not trip this.
if ! response="$(curl "${curl_args[@]}" "$UPSTREAM_API")"; then
    echo "error: could not reach the GitHub releases API ($UPSTREAM_API)" >&2
    exit 1
fi

latest="$(printf '%s' "$response" | grep -m1 '"tag_name"' | cut -d'"' -f4)"
if [ -z "$latest" ]; then
    echo "error: no tag_name in the releases API response — the response shape changed" >&2
    exit 1
fi

echo "declared tool-current: $declared"
echo "upstream latest stable: $latest"

if [ "$declared" != "$latest" ]; then
    cat >&2 <<EOF
error: upstream libjpeg-turbo $latest has shipped; our current-parity oracle is
       still $declared.

       Triage it rather than bumping in place (P4-130):
         * read the release notes for behaviour changes that touch tracked gaps;
         * decide whether $latest replaces $declared as tool-current or earns a
           third leg;
         * update docs/oracle_versions.tsv, the workflow pins that install it,
           and docs/FEATURE_PARITY.md's statement of what each gate proves.

       tests/oracle_version_pins.rs will fail until the manifest and the
       workflow pins agree again, so a half-finished bump cannot land quietly.
EOF
    exit 1
fi

echo "OK: the current-parity oracle matches upstream's latest stable release."
