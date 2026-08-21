<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# CLAUDE.md — repeatable pattern for the v5 self-contained flow

For an AI assistant (Claude Code) driving the **v5 self-contained** model flow with `tt-model`:
take a model brought up on `tt-metal-community`, package it, publish it, and serve it anywhere with
just a TT card. **This file is about *using* the flow. To change the tool's code, follow
[AGENTS.md](AGENTS.md) instead** (draft PR, one concern, regression test, never push to `main`).

**Scope: v5 self-contained only.** Do not route users down the v4 kernels-less or legacy dispatch
paths for new work. The full recipe with commands is [docs/E2E_RECIPE.md](docs/E2E_RECIPE.md); the
design is [docs/self_contained_packages.md](docs/self_contained_packages.md).

## The canonical sequence

```
bring up (tt-metal-community)  →  package  →  push  →  pull  →  serve  →  curl
```

1. **package** (producer): `tt-model package <org>/<name> --from-metal <dir> --wheels-dir <dir>
   --arch <isa> --arch-name <HFArch> --main-class <module:Class> --weights <hf-id> --mesh <mesh>
   --vendor-deps --repair`
2. **pull** (consumer): `tt-model pull <org>/<name>`
3. **serve** (consumer): `tt-model serve <org>/<name> [--port N] [--print]`
4. **verify**: `curl .../v1/chat/completions` → coherent text.

## Invariants — never violate these when helping a user

1. **tt-model alone must do the whole job.** Only prereq for a consumer is a TT card + firmware.
   Never introduce a step that needs tt-cli, a host tt-metal, or a host vLLM.
2. **The folder is the wall.** After `pull`, serving uses nothing outside the install directory
   except the TT device + system libc. The interpreter (`.python/`), venv, engine, and caches all
   live inside. Do not suggest pointing the model at a shared/system cache to "fix" something.
3. **Weights are a pointer** (`--weights <hf-id>`), never embedded in the bundle.
4. **The engine is what's on the box** — the author's `ttnn` wheel (kernels compiled in), made
   portable with `auditwheel --repair`. Never substitute a stock/pinned wheel.
5. **glibc floor is real.** Build/repair the engine wheel on the **oldest** target (Ubuntu 22.04,
   glibc 2.35) to serve both 22.04 and 24.04. A wheel repaired on 24.04 is 24.04-only.
6. **Serving contract = the plugin's `EXTRA_MODELS_DIR`**: `vllm_metadata.json` in a per-model
   *subfolder* (`vllm_models/<name>/`), not the bundle root.

## Verification checkpoints — do not claim success without them

- **After package:** the bundle has `wheels/` (incl. a `manylinux_*` ttnn wheel), `metal/`,
  `vllm_models/<name>/vllm_metadata.json`, `install.sh`, `run.sh`, `tt_kernel_manifest.json`.
- **After pull:** `install.sh` succeeded; `<install>/venv/bin/python` exists.
- **After serve:** the log reaches **`Application startup complete`** (model load + JIT warmup is
  minutes on a single chip — wait, don't declare failure early).
- **Only then** run the `curl` and confirm the text is coherent. Report the real result; if a step
  failed, say so with the output.

## Gotchas the flow encodes (each is a real past failure)

- Locate ttnn via `importlib.util.find_spec`, never `import ttnn`, when computing `LD_PRELOAD` — the
  import is what the preload fixes (glibc static-TLS). Prefer `ttnn.libs/_ttnncpp*.so` (the repaired
  copy) over `build/lib/`.
- `run.sh` must `export HF_MODEL` (the adapter reads it from env, not vLLM `--model`) and emit
  `--max_num_seqs` + `--block_size` (the TT backend rejects vLLM's 256/None defaults).
- Single-chip: fabric off + `TT_METAL_VISIBLE_DEVICES=0`.
- Do **not** set `VLLM_PLUGINS` — it's an allow-list that silently suppresses the model's
  tool/reasoning-parser plugins.
- Tool calling: the manifest must declare `capabilities.tool_parser`; the launcher emits
  `--enable-auto-tool-choice --tool-call-parser <name>` (vLLM normalizes `_`→`-`, so `--tool_parser`
  is the nonexistent `--tool-parser`).
- `serve` passes unknown args through to vLLM (e.g. `--port 8001`); `--print` echoes the resolved
  command+env without launching.

## When something's wrong

Prefer a clear, actionable message over a silent workaround. If `pull` refuses (glibc / interpreter
/ arch), the fix is to **repackage on the right OS**, not to force past the gate. Consult the
Troubleshooting table in [docs/E2E_RECIPE.md](docs/E2E_RECIPE.md).
