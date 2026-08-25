#!/usr/bin/env bash
# Bootstrap only: put `tt-model` on PATH, then hand off to `tt-model install`.
#
# Everything this script used to do — resolving the target venv, cloning the Tenstorrent
# vLLM fork, installing the serving layers, and verifying with `doctor` — now lives in the
# CLI, where it can reuse tt-model's own environment detection and render progress instead
# of ~400 lines of raw pip output. Run `tt-model install` directly once you have tt-model;
# this file exists for the one case that cannot: a fresh clone with no tt-model yet.
#
# Usage:
#   scripts/install.sh [any tt-model install flags]
#
# All arguments are forwarded verbatim, so:
#   scripts/install.sh --help
#   scripts/install.sh --venv <tt-metal>/python_env
#   scripts/install.sh --allow-no-ttnn
#
# Exit codes are `tt-model install`'s:
#   0  installed, and the toolchain is adequate
#   1  preflight failed — NOTHING was installed
#   2  usage error
#   3  installed, but the toolchain is NOT adequate
#
# Override the bootstrap interpreter with PYTHON=/path/to/python3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$(command -v python3 || true)}"

if [ -z "$PY" ]; then
  echo "ERROR: no python3 on PATH. Install Python 3.9+ (or set PYTHON=/path/to/python3)." >&2
  exit 1
fi

# Prefer an already-installed tt-model so a second run costs nothing. `-m tt_kernel.cli`
# rather than the console script: on a fresh clone the entry point may not be on PATH yet
# even though the package imports fine.
if ! "$PY" -c "import tt_kernel" >/dev/null 2>&1; then
  echo ">> Bootstrapping tt-model into $PY"
  "$PY" -m pip install -q -e "$REPO_ROOT"
fi

exec "$PY" -m tt_kernel.cli install "$@"
