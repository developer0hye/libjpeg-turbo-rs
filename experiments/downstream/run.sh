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

variant="${VARIANT:-default}"

# A clean slate for every flag that could change what is measured: the
# report's claim is "stock profile, no RUSTFLAGS" unless a variant says
# otherwise, so nothing inherited from the caller's shell may leak in.
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS
unset CARGO_PROFILE_RELEASE_LTO CARGO_PROFILE_RELEASE_CODEGEN_UNITS CARGO_PROFILE_RELEASE_OPT_LEVEL
unset CARGO_BUILD_TARGET
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
out_dir="${DOWNSTREAM_OUT_DIR:-$work_dir/report}"
mkdir -p "$out_dir"
out_dir=$(cd "$out_dir" && pwd -P)

# Cargo merges `.cargo/config.toml` from every ancestor of the build
# directory (https://doc.rust-lang.org/cargo/reference/config.html). One in an
# ancestor of the work dir would silently change the build, so refuse. The
# one in CARGO_HOME applies to every build on the machine, downstream ones
# included, so it is recorded in the report instead.
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
cargo_home_real=$(cd "$cargo_home" 2>/dev/null && pwd -P || echo "$cargo_home")
parent_configs=""
dir="$work_dir"
while :; do
  for name in config.toml config; do
    if [ -f "$dir/.cargo/$name" ] && [ "$(cd "$dir/.cargo" && pwd -P)" != "$cargo_home_real" ]; then
      parent_configs="$parent_configs $dir/.cargo/$name"
    fi
  done
  [ "$dir" = "/" ] && break
  dir=$(dirname "$dir")
done
[ -z "$parent_configs" ] || die "cargo config in an ancestor of $work_dir:$parent_configs"
home_configs=""
for name in config.toml config; do
  if [ -f "$cargo_home/$name" ]; then
    home_configs="$home_configs $cargo_home/$name"
  fi
done

consumer="$work_dir/consumer"
mkdir -p "$consumer"
# Copy without any stray target/ a developer may have built in-tree.
(cd "$consumer_src" && tar cf - --exclude ./target .) | (cd "$consumer" && tar xf -)
case "$repo_root" in
  *'|'* | *'&'*) die "repository path contains a character the substitution cannot carry: $repo_root" ;;
esac
sed "s|@CANDIDATE@|$repo_root|g" "$consumer/Cargo.toml" >"$consumer/Cargo.toml.new"
mv "$consumer/Cargo.toml.new" "$consumer/Cargo.toml"
if grep -q '@CANDIDATE@' "$consumer/Cargo.toml"; then
  die "placeholder substitution failed"
fi
cp "$consumer/Cargo.lock" "$work_dir/Cargo.lock.committed"

candidate_sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)
candidate_dirty=$(git -C "$repo_root" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')

echo "run.sh: variant=$variant work=$work_dir candidate=$repo_root@$candidate_sha" >&2

# --locked: the committed lock is the reviewed dependency set. If the
# candidate's dependencies changed so that the lock no longer fits, cargo
# refuses here rather than resolving fresh crates from the network — see the
# README's "Regenerating the lock".
build_started=$(now)
(cd "$consumer" && CARGO_TARGET_DIR="$work_dir/target-main" cargo build --release --locked --bin downstream-consumer) ||
  die "cargo build --release --locked failed (if the lock would change, regenerate it as the README describes)"
build_finished=$(now)
binary="$work_dir/target-main/release/downstream-consumer"
[ -x "$binary" ] || die "binary not produced at $binary"
cmp -s "$consumer/Cargo.lock" "$work_dir/Cargo.lock.committed" || die "Cargo.lock changed during a --locked build"

build_info="$work_dir/build-info.txt"
{
  echo "variant=$variant"
  echo "profile=release (Cargo default; the consumer manifest has no [profile] section)"
  echo "CARGO_PROFILE_RELEASE_LTO=${CARGO_PROFILE_RELEASE_LTO:-<unset: Cargo default lto=false>}"
  echo "RUSTFLAGS=${RUSTFLAGS:-<unset>}"
  echo "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-<unset: one job per CPU>}"
  echo "cargo=$(cargo -V)"
  echo "cargo_home_config=${home_configs:-<none>}"
  echo "candidate_sha=$candidate_sha"
  echo "candidate_tracked_changes=$candidate_dirty"
  echo "work_dir=$work_dir"
  echo "clean_build_seconds=$(elapsed "$build_started" "$build_finished") (all dependencies, fresh target dir)"
  echo "binary_bytes=$(file_bytes "$binary") (downstream-consumer, unstripped)"
} >"$build_info"

# Size probes: one package, features select one backend, each built from a
# fresh target dir so its build time is that backend's clean build cost.
if [ "${SKIP_PROBES:-0}" = "1" ]; then
  echo "size_probes=skipped (SKIP_PROBES=1)" >>"$build_info"
else
  none_bytes=""
  for probe in none candidate zune; do
    case "$probe" in
      none) features="" ;;
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

"$binary" --repo "$repo_root" --out-dir "$out_dir" --build-info "$build_info" \
  --lockfile "$consumer/Cargo.lock" "$@"
cp "$build_info" "$out_dir/build-info.txt"
echo "run.sh: report in $out_dir" >&2
