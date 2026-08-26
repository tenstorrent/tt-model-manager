<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# Thin (v6) model packages — DRAFT

> **Status: draft / scaffold.** This reflects the plan in **issue #29**. It becomes fully installable
> once the **models wheel** is published. That work is in progress upstream as **`tt-metal-models`**
> — [tenstorrent/tt-metal#54340](https://github.com/tenstorrent/tt-metal/pull/54340) — which packages
> the whole `models/` tree (**including `tt_transformers`**) for pip/apt/dnf and **pins `ttnn` exactly**
> (`tt-metal-models==X` ⇒ `ttnn==X`). So a thin bundle pins **one** dep (`tt-metal-models`) and the
> matching `ttnn` comes transitively — it likely subsumes a separate "TTTv2" wheel. Until it lands,
> the generated `requirements.txt` pins `ttnn` directly (it's on PyPI). See #29 for the full design.

A **thin (v6) bundle** keeps the self-contained wall between models — its own uv-managed venv — but
builds that venv from **pip dependency pins + a tiny models wheel** instead of embedding the full
platform. There is **no embedded `ttnn` wheel and no `metal/` tree**.

## What a thin bundle contains
```
<org>/<model>/
  tt_kernel_manifest.json     # schema_version "6"; carries a `deps` block (below)
  model.py                    # the runner; `--main-class module:Class` resolves it (PYTHONPATH=$HERE)
  requirements.txt            # pip pins: tt-metal-models (incl. tt_transformers; pins ttnn exactly) · custom ops · ...
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

## Author a thin bundle — from a "works on my box" model

A thin package is not the engine; it's a thin description of *your* model plus the recipe to rebuild
its venv anywhere. "Making the package" = capturing the exact recipe your working box used.

### What "working on my box" must already include → where it lands
| You have (working box) | Becomes (in the package) |
|---|---|
| `model.py` — your runner, built on the `tt_transformers` blocks (from `tt-metal-models`) and/or calling `ttnn.generic_op` for custom ops | shipped at the bundle root |
| the **exact dep versions** you ran with — `tt-metal-models==…` (pulls the matching `ttnn`), any custom-op wheel | `requirements.txt` pins |
| *(if you wrote custom ops)* a **`generic_op` wheel** you built | `custom_ops/` + a pin |
| the **serving entrypoint** — the HF architecture name + the `module:Class` the vLLM plugin loads | `vllm_metadata.json` (`arch` → `main_class`) |
| the **weights** (an HF repo id) + the **serving knobs** you validated (mesh, max_num_seqs, block_size) | manifest pointer + `resources`/`mesh` |

The key discipline: **pin what actually worked** — `pip freeze` in your working venv gives the real
`tt-metal-models` (and, until it lands, `ttnn`) version to put in `requirements.txt`.

### Steps
1. **Make `model.py` importable by its class path.** Class `QwenForCausalLM` in `model.py` → the
   entrypoint is `model:QwenForCausalLM` (module = filename without `.py`; at serve time `PYTHONPATH`
   is the bundle root).
2. **Write `requirements.txt`** with the versions your box ran:
   ```
   # tt-metal-models==<X>       # the models tree (incl. tt_transformers); pins ttnn==<X> exactly
   #                            # (upstream tt-metal#54340) — pulls the matching ttnn transitively
   ttnn==0.77.0                 # engine (PyPI today; pin directly until tt-metal-models lands)
   my_model_ops==0.1            # your generic_op custom-op wheel (if any)
   ```
   SFPI + firmware are external box deps — **not** in `requirements.txt`. (Omit `--requirements` and
   `package-thin` writes this as a template with the `tt-metal-models` TODO pin.)
3. **(Custom ops only)** put your built `generic_op` wheel in a folder, e.g.
   `./custom_ops/my_model_ops-0.1-*.whl`; `model.py` calls `ttnn.generic_op(...)` to use it.
4. **Run `package-thin`:**
   ```bash
   tt-model package-thin <org>/<model> \
     --model-py ./model.py \
     --requirements ./requirements.txt \      # your pins (or omit for the #29 TODO template)
     --wheels-dir ./custom_ops \              # optional: your generic_op wheel(s)
     --arch blackhole \
     --arch-name QwenForCausalLM --main-class model:QwenForCausalLM \
     --weights Qwen/Qwen3-4B \                # pointer, never embedded
     --mesh P150 --max-num-seqs 32 --block-size 64 \
     --out ./bundle                           # stage locally (omit + pass <org>/<model> to push)
   ```
5. **Result** — the bundle layout shown above (`model.py` + `requirements.txt` + `custom_ops/` +
   `vllm_models/<name>/vllm_metadata.json` + manifest + `install.sh`/`run.sh`; no `wheels/`, no `metal/`).
6. **Round-trip** — on any card + firmware + SFPI box: `tt-model pull <org>/<model>` builds the venv
   from `requirements.txt` and fetches the weights; `tt-model serve <org>/<model>` launches it.

**Required from you:** `model.py` + a `requirements.txt` of real pins + *(optional)* a `generic_op`
wheel + the entrypoint (`--arch-name`/`--main-class`) + a weights repo id. Everything else is generated.

## Testing it in the lab (today)
1. `tt-model package-thin ... --out /tmp/thin`
2. Edit `/tmp/thin/requirements.txt` — pin the real `ttnn` now, and `tt-metal-models` once tt-metal#54340 publishes.
3. `bash /tmp/thin/install.sh` (builds the venv from the pins; needs SFPI on the box).
4. `bash /tmp/thin/run.sh` (serves), then `curl` the endpoint — or `tt-model pull`/`serve` from HF.
