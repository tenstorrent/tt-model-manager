#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tt-model container entrypoint. The serve command arrives as "$@" (composed on the
# host from the manifest's serve profile); env like MESH_DEVICE / HF_MODEL arrives via
# docker run -e. This script only prepares the runtime dirs and execs.
set -euo pipefail

# The JIT kernel/trace cache. /cache is a host bind mount so the ~10 min compile is
# paid once, not per boot.
export TT_METAL_CACHE="${TT_METAL_CACHE:-/cache}"
mkdir -p "$TT_METAL_CACHE" 2>/dev/null || true

# Inspector/watcher/model_cache write RELATIVE to cwd (and TT_METAL_LOGS_PATH), so cwd
# must be writable — that is why WORKDIR is /home/tt/work, not the metal tree.
export TT_METAL_LOGS_PATH="${TT_METAL_LOGS_PATH:-$HOME/work/logs}"
mkdir -p "$TT_METAL_LOGS_PATH"

# VLLM_PLUGINS is an ALLOW-list: setting it (to anything) suppresses the
# vllm.general_plugins group and silently kills the model's tool/reasoning parsers.
unset VLLM_PLUGINS

cd "$HOME/work"

# exec so the server is PID 1 and receives docker stop's SIGTERM: a clean shutdown
# closes the mesh; a SIGKILL leaves the devices needing tt-smi -r before reopening.
exec "$@"
