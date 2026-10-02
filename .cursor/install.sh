#!/usr/bin/env bash
# Idempotent Cloud Agent setup for EasyHybrid.jl.
# Installs the pinned Julia toolchain via juliaup and instantiates the
# package environment (dependencies + precompilation).
set -euo pipefail

JULIA_CHANNEL="1.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIAUP_BIN="$HOME/.juliaup/bin"

if [ ! -x "$JULIAUP_BIN/juliaup" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel "$JULIA_CHANNEL"
fi

export PATH="$JULIAUP_BIN:$PATH"

juliaup add "$JULIA_CHANNEL" >/dev/null 2>&1 || true
juliaup default "$JULIA_CHANNEL" >/dev/null 2>&1 || true

# Expose julia on the global PATH so non-login shells can find it too.
if [ ! -e /usr/local/bin/julia ] && command -v sudo >/dev/null 2>&1; then
    sudo ln -sf "$JULIAUP_BIN/julia" /usr/local/bin/julia || true
fi

# Instantiate + precompile the package dependencies.
julia --project="$REPO_ROOT" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "EasyHybrid.jl environment ready ($(julia --version))."
