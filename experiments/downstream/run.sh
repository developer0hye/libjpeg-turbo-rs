#!/usr/bin/env bash
# Downstream-consumer benchmark runner (P4-214, issue #640).
#
# Builds experiments/downstream/consumer the way an application would build
# it — a separate crate outside this repository, Cargo's stock `release`
# profile, no RUSTFLAGS — and runs it against this checkout as the candidate.
# See experiments/downstream/README.md.
#
# Usage:
#   experiments/downstream/run.sh [harness args...]
#     e.g. experiments/downstream/run.sh --iterations 30 --warmup 5
#          experiments/downstream/run.sh --smoke
#   experiments/downstream/run.sh --check
#     lint and unit-test the copied consumer (cargo fmt --check, clippy
#     -D warnings, cargo test), then exit without benchmarking
#
# Environment:
#   VARIANT=default|thin-lto|fat-lto|native   build variant (default: default)
#   DOWNSTREAM_WORK_DIR=<dir>   where the consumer is copied and built; must
#                               be outside the repository and empty or absent
#                               (default: a fresh `mktemp -d`)
#   DOWNSTREAM_OUT_DIR=<dir>    report directory (default: $WORK/report)
#   SKIP_PROBES=1               skip the per-backend size-probe builds
#   CARGO_BUILD_JOBS            passed through to cargo and recorded
#
# Footprint: the main build and the six probe builds each use their own
# target directory (several hundred MB together). They are deleted once the
# binary and the sizes are recorded; the work dir keeps the copied consumer,
# Cargo.lock.committed, the binary, build-info.txt and the report.
#
# Only bash 3.2 features are used: it is what macOS ships, locally and on the
# hosted macos runner.
set -euo pipefail

die() {
  echo "run.sh: $*" >&2
  exit 1
}

now() {
  # Sub-second wall clock without GNU date (macOS `date` has no %N).
  python3 -c 'import time; print("%.3f" % time.time())' 2>/dev/null || date +%s
}

elapsed() {
  python3 -c "import sys; print('%.1f' % (float(sys.argv[2]) - float(sys.argv[1])))" "$1" "$2" 2>/dev/null ||
    echo $(($2 - $1))
}

file_bytes() {
  wc -c <"$1" | tr -d ' '
}

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)
consumer_src="$script_dir/consumer"
[ -f "$consumer_src/Cargo.toml" ] || die "consumer crate not found at $consumer_src"
[ -f "$consumer_src/Cargo.lock" ] || die "consumer Cargo.lock missing: builds must be --locked"
[ -f "$repo_root/references/libjpeg-turbo/testimages/testorig.jpg" ] ||
  die "references/libjpeg-turbo is not checked out (git submodule update --init references/libjpeg-turbo)"

mode="bench"
if [ "${1:-}" = "--check" ]; then
  mode="check"
  shift
fi
variant="${VARIANT:-default}"

# A clean slate for every setting that could change what is measured: the
# report's claim is "stock profile, no RUSTFLAGS" unless a variant says
# otherwise, so nothing inherited from the caller's shell may leak in. The
# names are discovered rather than listed, so a CARGO_PROFILE_RELEASE_* or
# CARGO_TARGET_<triple>_RUSTFLAGS nobody thought of is cleared too.
# CARGO_BUILD_JOBS is kept: it changes build time, not the binary, and is
# recorded. (grep -E, not sed: BSD sed has no \| alternation.)
scrubbed=$(env | grep -E '^(CARGO_PROFILE_[A-Z0-9_]*|CARGO_BUILD_[A-Z0-9_]*|CARGO_TARGET_[A-Z0-9_]*|CARGO_INCREMENTAL|RUSTC_WRAPPER|RUSTC_WORKSPACE_WRAPPER|RUSTFLAGS|CARGO_ENCODED_RUSTFLAGS|RUSTDOCFLAGS|CARGO_ENCODED_RUSTDOCFLAGS)=' |
  cut -d= -f1 | grep -vx 'CARGO_BUILD_JOBS' || true)
for name in $scrubbed; do
  unset "$name"
done
case "$variant" in
  default) ;;
  thin-lto) export CARGO_PROFILE_RELEASE_LTO=thin ;;
  fat-lto) export CARGO_PROFILE_RELEASE_LTO=fat ;;
  native) export RUSTFLAGS="-Ctarget-cpu=native" ;;
  *) die "unknown VARIANT '$variant' (default|thin-lto|fat-lto|native)" ;;
esac

if [ -n "${DOWNSTREAM_WORK_DIR:-}" ]; then
  mkdir -p "$DOWNSTREAM_WORK_DIR"
  [ -z "$(ls -A "$DOWNSTREAM_WORK_DIR")" ] ||
    die "DOWNSTREAM_WORK_DIR=$DOWNSTREAM_WORK_DIR is not empty; the build time is only a clean-build time in a fresh directory"
  work_dir=$(cd "$DOWNSTREAM_WORK_DIR" && pwd -P)
else
  work_dir=$(cd "$(mktemp -d)" && pwd -P)
fi
case "$work_dir/" in
  "$repo_root/"*) die "work dir $work_dir is inside the repository; its .cargo/config.toml would apply" ;;
esac

consumer="$work_dir/consumer"
mkdir -p "$consumer"
# Copy without any stray target/ a developer may have built in-tree.
(cd "$consumer_src" && tar cf - --exclude ./target .) | (cd "$consumer" && tar xf -)
# The path lands inside a TOML basic string and a sed replacement.
case "$repo_root" in
  *'|'* | *'&'* | *\\* | *'"'*) die "repository path contains a character the substitution cannot carry: $repo_root" ;;
esac
sed "s|@CANDIDATE@|$repo_root|g" "$consumer/Cargo.toml" >"$consumer/Cargo.toml.new"
mv "$consumer/Cargo.toml.new" "$consumer/Cargo.toml"
if grep -q '@CANDIDATE@' "$consumer/Cargo.toml"; then
  die "placeholder substitution failed"
fi
# The report says "stock release profile"; check rather than assume.
if grep -Eq '^[[:space:]]*\[profile' "$consumer/Cargo.toml"; then
  die "the consumer manifest declares a [profile] section; the stock profile is the point"
fi
cp "$consumer/Cargo.lock" "$work_dir/Cargo.lock.committed"

# Cargo merges `.cargo/config.toml` from the build directory and every
# ancestor (https://doc.rust-lang.org/cargo/reference/config.html). One there
# would silently change the build, so refuse. CARGO_HOME's config applies to
# every build on the machine, downstream ones included: refuse it if it
# changes code generation, otherwise record its shape (table and key names,
# values redacted; see build-info below).
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
cargo_home_real=$(cd "$cargo_home" 2>/dev/null && pwd -P || echo "$cargo_home")
parent_configs=""
dir="$consumer"
while :; do
  for name in config.toml config; do
    if [ -f "$dir/.cargo/$name" ] && [ "$(cd "$dir/.cargo" && pwd -P)" != "$cargo_home_real" ]; then
      parent_configs="$parent_configs $dir/.cargo/$name"
    fi
  done
  [ "$dir" = "/" ] && break
  dir=$(dirname "$dir")
done
[ -z "$parent_configs" ] || die "cargo config in $consumer or an ancestor:$parent_configs"
home_configs=""
for name in config.toml config; do
  if [ -f "$cargo_home/$name" ]; then
    if grep -Eqi 'profile|rustflags|rustc-wrapper|rustc_wrapper|rustc-workspace-wrapper|^[[:space:]]*\[target' "$cargo_home/$name"; then
      die "$cargo_home/$name sets profile, rustflags, a rustc wrapper or [target] options; they would apply to this build"
    fi
    home_configs="$home_configs $cargo_home/$name"
  fi
done

if [ "$mode" = "check" ]; then
  # Lint and test the consumer as copied (path substituted), then stop.
  (
    cd "$consumer"
    export CARGO_TARGET_DIR="$work_dir/target-check"
    cargo fmt --check
    cargo clippy --locked --release --all-targets -- -D warnings
    cargo test --locked --release
  ) || die "consumer check failed"
  rm -rf "$work_dir/target-check"
  echo "run.sh: consumer check passed" >&2
  exit 0
fi

candidate_sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)
candidate_dirty=$(git -C "$repo_root" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')
# The harness binary itself moves same-run ratios: two consumer builds of one
# library differed by 5 % and 30 % on two parity rows (BUDGETS.md), so budgets.py only
# compares reports built from the same consumer sources. Hash the consumer
# directory as copied: the working tree, untracked files included and
# target/ excluded, before the path substitution, file names included.
consumer_source_sha256=$(cd "$consumer_src" &&
  find . -path ./target -prune -o -type f -print | LC_ALL=C sort |
  while IFS= read -r file; do
    printf '%s %s\n' "$(shasum -a 256 <"$file" | cut -d' ' -f1)" "$file"
  done | shasum -a 256 | cut -d' ' -f1)

echo "run.sh: variant=$variant work=$work_dir candidate=$repo_root@$candidate_sha" >&2

# --locked: the committed lock is the reviewed dependency set. If the
# candidate's dependencies changed so that the lock no longer fits, cargo
# refuses here rather than resolving fresh crates from the network — see the
# README's "Regenerating the lock".
out_dir="${DOWNSTREAM_OUT_DIR:-$work_dir/report}"
mkdir -p "$out_dir"
out_dir=$(cd "$out_dir" && pwd -P)

# Downloads are not build time: fetch the locked crates first.
(cd "$consumer" && cargo fetch --locked) || die "cargo fetch --locked failed"
build_started=$(now)
(cd "$consumer" && CARGO_TARGET_DIR="$work_dir/target-main" cargo build --release --locked --bin downstream-consumer) ||
  die "cargo build --release --locked failed (if the lock would change, regenerate it as the README describes)"
build_finished=$(now)
binary="$work_dir/downstream-consumer"
cp "$work_dir/target-main/release/downstream-consumer" "$binary" || die "binary not produced"
cmp -s "$consumer/Cargo.lock" "$work_dir/Cargo.lock.committed" || die "Cargo.lock changed during a --locked build"

build_info="$work_dir/build-info.txt"
{
  echo "variant=$variant"
  echo "profile=release; consumer manifest has no [profile] section (checked)"
  echo "CARGO_PROFILE_RELEASE_LTO=${CARGO_PROFILE_RELEASE_LTO:-<unset: Cargo default lto=false>}"
  echo "RUSTFLAGS=${RUSTFLAGS:-<unset>}"
  echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-<unset: one job per CPU>}"
  echo "scrubbed_env=$(printf '%s' "$scrubbed" | tr '\n' ' ')"
  echo "cargo=$(cd "$consumer" && cargo -V)"
  echo "cargo_home_config=${home_configs:-<none>}"
  # The report is uploaded as an artifact, and a cargo config can hold
  # registry tokens, proxy URLs with credentials, or private index URLs. So
  # only its *shape* is recorded: table names and key names, with every value
  # redacted except two harmless build-affecting ones ([build] jobs and
  # [net] offline). Lines that are not `[table]` or `key = ...` (continuation
  # lines of a multi-line array or string) are skipped, never echoed.
  for config in $home_configs; do
    awk '
      /^[[:space:]]*(#|$)/ { next }
      /^[[:space:]]*\[/ {
        table = $0
        sub(/^[[:space:]]*\[+[[:space:]]*/, "", table)
        sub(/[[:space:]]*\]+.*$/, "", table)
        print "cargo_home_config_table=" table
        next
      }
      /^[[:space:]]*[A-Za-z0-9_."-]+[[:space:]]*=/ {
        key = $0
        sub(/^[[:space:]]*/, "", key)
        sub(/[[:space:]]*=.*$/, "", key)
        full = (table == "" ? key : table "." key)
        value = "<redacted>"
        if (full == "build.jobs" || full == "net.offline") {
          value = $0
          sub(/^[^=]*=[[:space:]]*/, "", value)
          sub(/[[:space:]]*#.*$/, "", value)
        }
        print "cargo_home_config_key=" full " = " value
      }
    ' "$config"
  done
  # Credentials by name (CARGO_REGISTRY_TOKEN and kin) and any value that is
  # a URL with userinfo (`scheme://user:pass@host`) are dropped.
  env | grep -E '^(CARGO_|RUST)' | grep -v '^CARGO_HOME=' |
    grep -Ev '^[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*=' |
    grep -Ev '=.*://.*@' | sort | sed 's|^|env_|' || true
  echo "candidate_sha=$candidate_sha"
  echo "candidate_tracked_changes=$candidate_dirty"
  echo "consumer_source_sha256=$consumer_source_sha256"
  echo "work_dir=$work_dir"
  echo "clean_build_seconds=$(elapsed "$build_started" "$build_finished") (all dependencies, fresh target dir, downloads excluded)"
  echo "binary_bytes=$(file_bytes "$binary") (downstream-consumer, unstripped)"
} >"$build_info"

# Size probes: one package, features select one backend, each built from a
# fresh target dir so its build time is that backend's clean build cost.
if [ "${SKIP_PROBES:-0}" = "1" ]; then
  echo "size_probes=skipped (SKIP_PROBES=1)" >>"$build_info"
else
  none_bytes=""
  for probe in none baseline candidate adapter image zune; do
    case "$probe" in
      none) features="" ;;
      image) features="--features image-builtin" ;;
      *) features="--features $probe" ;;
    esac
    probe_started=$(now)
    # shellcheck disable=SC2086 # $features is intentionally split
    (cd "$consumer" && CARGO_TARGET_DIR="$work_dir/target-probe-$probe" \
      cargo build --release --locked --no-default-features $features --bin "probe-$probe") ||
      die "probe-$probe build failed"
    probe_finished=$(now)
    probe_binary="$work_dir/target-probe-$probe/release/probe-$probe"
    bytes=$(file_bytes "$probe_binary")
    [ "$probe" = "none" ] && none_bytes=$bytes
    echo "probe_${probe}_bytes=$bytes (contribution over probe-none: $((bytes - none_bytes)))" >>"$build_info"
    echo "probe_${probe}_clean_build_seconds=$(elapsed "$probe_started" "$probe_finished")" >>"$build_info"
  done
fi

# The build trees are the bulk of the footprint and nothing below needs them.
rm -rf "$work_dir"/target-main "$work_dir"/target-probe-*

# Run from the consumer directory so the harness's own `rustc -Vv` resolves
# the same toolchain (rustup overrides are per directory) as the build did.
(cd "$consumer" && "$binary" --repo "$repo_root" --out-dir "$out_dir" --build-info "$build_info" \
  --lockfile "$consumer/Cargo.lock" "$@")
cp "$build_info" "$out_dir/build-info.txt"
echo "run.sh: report in $out_dir" >&2
