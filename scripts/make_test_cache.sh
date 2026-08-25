#!/usr/bin/env bash
# Deprecated shim — the generator now lives in the CLI as `tt-model dev make-test-cache`.
#
# It moved because the layout it fabricates has to track cache.resolve_out_root, and a
# shell script in another directory drifts from that silently.
#
# Usage (unchanged): scripts/make_test_cache.sh [ROOT] [BUILD_KEY] [--with-runner]
set -euo pipefail

echo "note: scripts/make_test_cache.sh is deprecated; use \`tt-model dev make-test-cache\`." >&2

PY="${PYTHON:-$(command -v python3 || true)}"
[ -n "$PY" ] || { echo "ERROR: no python3 on PATH." >&2; exit 1; }

ARGS=()
for arg in "$@"; do
  [ "$arg" = "--with-runner" ] && ARGS+=("--with-runner") || ARGS+=("$arg")
done

exec "$PY" -m tt_kernel.cli dev make-test-cache "${ARGS[@]}"
