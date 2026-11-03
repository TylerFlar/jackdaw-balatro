#!/usr/bin/env bash
# Profile-guided optimisation (PGO) of the jackdaw._core extension.
#
#   scripts/pgo.sh build [OUT_DIR]          the whole pipeline here: an instrumented wheel, the
#                                           training workload, then the optimised wheel in
#                                           OUT_DIR (default: dist/)
#   scripts/pgo.sh train WHEEL_DIR PROFDATA run the training workload against the instrumented
#                                           wheel in WHEEL_DIR and merge its profile into
#                                           PROFDATA (the Wheels workflow builds both wheels
#                                           with maturin-action and runs this in between)
#
# The training workload is the `rollout` example (random and `--deep` policies) built with
# instrumentation, plus scripts/pgo_workload.py, which drives the instrumented wheel through
# BalatroGymnasiumEnv and BalatroVectorEnv. The rollout example and the wheel share their
# crates' symbol names (same toolchain and target), so one merged profile covers both.
#
# PGO only changes how the compiler lays out and inlines code, never what it computes: the
# optimised build gives the same results (`rollout --verify` prints the same hash). No
# target-cpu flags, so the wheel runs on any CPU of its target.
#
# Needs rustup (it adds the llvm-tools component for llvm-profdata), Python 3.12+ and network
# access for the workload's dependencies (gymnasium, numpy). `uv` is used when present.
# Environment: PYTHON (default python3, else python) for the training venv;
# PGO_ROLLOUT_STEPS (default 400000) per rollout policy; PGO_PY_STEPS (default 100000).
set -euo pipefail

CALLER="$PWD"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() {
    echo "pgo.sh: $*" >&2
    exit 1
}

# The arguments' paths are relative to where the script was run from.
abspath() {
    case "$1" in
        /* | [A-Za-z]:*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$CALLER/$1" ;;
    esac
}

host_triple() {
    rustc -vV | sed -n 's/^host: //p'
}

# A path native Windows programs understand (Git Bash hands them /c/... paths otherwise).
native_path() {
    if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}

llvm_profdata() {
    rustup component add llvm-tools > /dev/null
    local bin
    bin="$(rustc --print sysroot)/lib/rustlib/$(host_triple)/bin/llvm-profdata"
    [ -x "$bin" ] || bin="$bin.exe"
    [ -x "$bin" ] || die "llvm-profdata not found next to $(rustc --version)"
    printf '%s\n' "$bin"
}

python_cmd() {
    if [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "$PYTHON"
    elif command -v python3 > /dev/null 2>&1; then
        echo python3
    else
        echo python
    fi
}

# train WHEEL_DIR PROFDATA
train() {
    local wheel_dir="$1" out="$2"
    local wheels=("$wheel_dir"/*.whl)
    [ -f "${wheels[0]}" ] || die "no wheel in $wheel_dir"
    [ "${#wheels[@]}" -eq 1 ] || die "more than one wheel in $wheel_dir"

    local work="$ROOT/target/pgo/train"
    rm -rf "$work"
    mkdir -p "$work/profraw"
    local profraw
    profraw="$(native_path "$work/profraw")"

    # The Python workload, against the instrumented wheel in a throwaway venv.
    local py
    if command -v uv > /dev/null 2>&1; then
        uv venv --quiet --python "$(python_cmd)" "$work/venv"
        py="$work/venv/bin/python"
        [ -x "$py" ] || py="$work/venv/Scripts/python.exe"
        uv pip install --quiet --python "$py" "${wheels[0]}"
    else
        "$(python_cmd)" -m venv "$work/venv"
        py="$work/venv/bin/python"
        [ -x "$py" ] || py="$work/venv/Scripts/python.exe"
        "$py" -m pip install --quiet "${wheels[0]}"
    fi
    LLVM_PROFILE_FILE="$profraw/wheel-%p-%m.profraw" \
        "$py" -I scripts/pgo_workload.py --steps "${PGO_PY_STEPS:-100000}"

    # The engine: the rollout example, instrumented the same way, for the target the
    # wheel is built for.
    local triple target_dir
    triple="$(host_triple)"
    target_dir="$ROOT/target/pgo/rollout"
    RUSTFLAGS="${RUSTFLAGS:-} -Cprofile-generate=$profraw" CARGO_TARGET_DIR="$target_dir" \
        cargo build --release --locked --target "$triple" -p jackdaw-env --example rollout
    local rollout="$target_dir/$triple/release/examples/rollout"
    [ -f "$rollout" ] || rollout="$rollout.exe"
    local steps="${PGO_ROLLOUT_STEPS:-400000}"
    LLVM_PROFILE_FILE="$profraw/rollout-%p-%m.profraw" "$rollout" "$steps"
    LLVM_PROFILE_FILE="$profraw/rollout-%p-%m.profraw" "$rollout" "$steps" --deep

    compgen -G "$work/profraw/wheel-*.profraw" > /dev/null || die "the wheel wrote no profile"
    mkdir -p "$(dirname "$out")"
    "$(llvm_profdata)" merge -o "$(native_path "$out")" "$profraw"
    echo "pgo.sh: merged the profile into $out"
}

# build [OUT_DIR]
build() {
    local out="$1"
    command -v uv > /dev/null 2>&1 || die "build needs uv (https://docs.astral.sh/uv/)"
    local triple profdata
    triple="$(host_triple)"
    profdata="$ROOT/target/pgo/jackdaw.profdata"
    # A target dir of its own, so the PGO builds don't evict the usual one; an explicit target
    # (as maturin-action passes) keeps the flags off build scripts and proc macros.
    export CARGO_TARGET_DIR="$ROOT/target/pgo/wheel"
    export CARGO_BUILD_TARGET="$triple"
    local base="${RUSTFLAGS:-}"

    rm -rf "$ROOT/target/pgo/instrumented"
    RUSTFLAGS="$base -Cprofile-generate=$(native_path "$ROOT/target/pgo/default")" \
        uv build --wheel --out-dir "$ROOT/target/pgo/instrumented"
    train "$ROOT/target/pgo/instrumented" "$profdata"
    RUSTFLAGS="$base -Cprofile-use=$(native_path "$profdata")" uv build --wheel --out-dir "$out"
}

case "${1:-}" in
    train)
        [ $# -eq 3 ] || die "usage: scripts/pgo.sh train WHEEL_DIR PROFDATA"
        train "$(abspath "$2")" "$(abspath "$3")"
        ;;
    build)
        [ $# -le 2 ] || die "usage: scripts/pgo.sh build [OUT_DIR]"
        build "$(abspath "${2:-dist}")"
        ;;
    *)
        die "usage: scripts/pgo.sh build [OUT_DIR] | scripts/pgo.sh train WHEEL_DIR PROFDATA"
        ;;
esac
