<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# Thin (v6) model packages — DRAFT

> **Status: draft / scaffold.** This reflects the plan in **issue #29**. It becomes fully installable
> once **TTTv2** (`tt-transformers` v2) and the **models wheel** are published so the requirements can
> pin real versions. Until then the generated `requirements.txt` carries TODO pins for those two
> (`ttnn` already resolves from PyPI). See #29 for the full design and open questions.

A **thin (v6) bundle** keeps the self-contained wall between models — its own uv-managed venv — but
builds that venv from **pip dependency pins + a tiny models wheel** instead of embedding the full
platform. There is **no embedded `ttnn` wheel and no `metal/` tree**.

## What a thin bundle contains
```
<org>/<model>/
  tt_kernel_manifest.json     # schema_version "6"; carries a `deps` block (below)
  model.py                    # the runner; `--main-class module:Class` resolves it (PYTHONPATH=$HERE)
  requirements.txt            # pip pins: ttnn (PyPI/team) · tt-transformers (TTTv2) · models wheel · ...
  custom_ops/                 # OPTIONAL: bundled generic_op custom-op wheel(s), added via --find-links
  vllm_models/<name>/vllm_metadata.json   # EXTRA_MODELS_DIR contract (arch -> main_class)
  install.sh  run.sh
  # weights: NOT embedded — HF pointer in the manifest
```

## Manifest `deps` block (schema 6)
- `python` — pinned interpreter (uv provisions it into the bundle)
- `requirements` — the pins file (default `requirements.txt`)
- `wheels_dir` — optional bundle dir of shipped wheels (e.g. `custom_ops`) → `--find-links`
- `model_dir` — where `model.py` lives (default the bundle root) → PYTHONPATH at serve

## Install / serve
`install.sh` builds the venv with uv: `uv venv --relocatable --python <pin>`, then
`uv pip install [--find-links custom_ops] -r requirements.txt`. `run.sh` wires the engine env
(`LD_PRELOAD` of `_ttnncpp.so`, `TT_METAL_HOME` at the installed `ttnn`, `EXTRA_MODELS_DIR`,
hermetic caches under the folder) and launches vLLM — `PYTHONPATH=$HERE` so `model.py` imports.
`tt-model pull` / `serve` route a thin bundle through the same install/serve path as a v5 fat one.

## Box prerequisites
A TT **card**, its **firmware/driver**, and **SFPI** (SFPI is a separate, externally-managed box
dependency — provisioned by tt-cli, or installed by the user on a bare box; it is **not** in `ttnn`
and **not** in the venv). No separate tt-metal install — the tt-metal runtime rides inside `ttnn`.

## Author a thin bundle
```bash
tt-model package-thin <org>/<model> \
  --model-py ./model.py \
  --requirements ./requirements.txt \      # your pins (or omit for the #29 TODO template)
  --wheels-dir ./custom_ops \              # optional: your generic_op wheel(s)
  --arch blackhole --arch-name QwenForCausalLM --main-class model:QwenForCausalLM \
  --weights Qwen/Qwen3-4B --mesh P150 \
  --out ./bundle                           # stage locally (omit + pass <org>/<model> to push)
```

## Testing it in the lab (today)
1. `tt-model package-thin ... --out /tmp/thin`
2. Edit `/tmp/thin/requirements.txt` — pin the real `ttnn` and, once they exist, TTTv2 + the models wheel.
3. `bash /tmp/thin/install.sh` (builds the venv from the pins; needs SFPI on the box).
4. `bash /tmp/thin/run.sh` (serves), then `curl` the endpoint — or `tt-model pull`/`serve` from HF.
