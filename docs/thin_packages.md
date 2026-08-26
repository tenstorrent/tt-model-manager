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
  vllm-overrides.txt          # numpy<2 / opencv pins for the empty-target vLLM build (see below)
  wheels/                     # bundled wheels installed BY PATH: vllm-tt-plugin (the vLLM
                              # integration) + any generic_op custom-op wheel [+ optional prebuilt vLLM wheel]
  vllm_models/<name>/vllm_metadata.json   # EXTRA_MODELS_DIR contract (arch -> main_class)
  install.sh  run.sh
  # weights: NOT embedded — HF pointer in the manifest
```

## Manifest `deps` block (schema 6)
- `python` — pinned interpreter (uv provisions it into the bundle)
- `requirements` — the pins file (default `requirements.txt`) — ttnn / tt-metal-models only
- `wheels` — bundle-relative wheels installed **by path** (vllm-tt-plugin, then any generic_op wheels)
- `wheels_dir` — the dir holding them (`wheels`), also put on `--find-links`
- `vllm` — the empty-target vLLM install step (below): `version`, `target_device` (`empty`),
  `overrides` (the pins file), `common_requirements` (optional bundled copy, else fetched),
  `wheel` (optional prebuilt empty-target wheel). `None` for a non-vLLM model.
- `model_dir` — where `model.py` lives (default the bundle root) → PYTHONPATH at serve

## Install / serve
`install.sh` builds the venv with uv (`uv venv --relocatable --python <pin>`), then installs in a
**load-bearing order**:

1. **Engine + models** — `uv pip install -r requirements.txt` (ttnn, and once published
   `tt-metal-models`). This lands `torch` + `numpy<2` FIRST so vLLM's deps resolve against them.
2. **vLLM core (empty target)** — this is **not** a pip pin. It is **stock upstream vLLM built with
   `VLLM_TARGET_DEVICE=empty`** (NOT the CUDA `vllm` on PyPI), exactly mirroring the plugin's
   [`docs/install-vllm-tt.sh`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/install-vllm-tt.sh):
   - `uv pip install --override vllm-overrides.txt -r <vLLM v0.25.1 common.txt>` — the overrides pin
     `opencv-python-headless==4.11.0.86` + `numpy>=1.24.4,<2` so ttnn's `numpy<2` is not bumped
     (opencv is vLLM's only numpy-2 puller and no TT-registered model uses its video path). The
     upstream `common.txt` is fetched from the pinned tag unless a copy is bundled.
   - `VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm vllm==0.25.1` — builds vLLM
     from source. (Or, if a prebuilt empty-target wheel is bundled, `uv pip install --no-deps <wheel>`.)
   The `tt` platform is supplied by the plugin at runtime; `VLLM_TARGET_DEVICE` is a **build-time**
   var only, never set at serve.
3. **Plugin + custom ops** — `uv pip install [--find-links wheels] <vllm-tt-plugin + generic_op wheels>`,
   installed by path AFTER vLLM. The plugin's `pyproject` omits `vllm` on purpose, so this never
   re-resolves (and never clobbers) the empty-target build.

`run.sh` wires the engine env (`LD_PRELOAD` of `_ttnncpp.so`, `TT_METAL_HOME` at the installed
`ttnn`, `EXTRA_MODELS_DIR`, hermetic caches under the folder) and launches vLLM — `PYTHONPATH=$HERE`
so `model.py` imports. `tt-model pull` / `serve` route a thin bundle through the same install/serve
path as a v5 fat one.

> **Why vLLM is a build, not a pin:** the plugin runs against a vLLM compiled for an *empty* device
> target so it can inject the `tt` platform out-of-tree. The PyPI `vllm` wheel is CUDA-built and
> would break that. A resolvable `vllm` pin anywhere (requirements, or the plugin's own deps) would
> silently uninstall the empty build and pull the CUDA wheel — which is exactly why the plugin's
> `pyproject` omits `vllm` and why we install it in its own `--no-deps` step.

> **On `run.sh`:** the serving launcher is a generated `run.sh`, and `tt-model serve` runs
> `bash <bundle>/run.sh`. It's a shell wrapper (not Python in `serve`) on purpose: `LD_PRELOAD` of
> `_ttnncpp.so` must be set **before** the interpreter starts, so it can't be done from inside
> tt-model's own process. `run.sh` also makes the bundle runnable without tt-model (`bash run.sh`);
> `tt-model serve` is the managed wrapper. The author doesn't write it — `package`/`package-thin`
> generate it; the author writes `model.py`.

## Serving front end — LLM today, other model types later (NOT in this draft)

The engine + bundle + install are **modality-agnostic** and shared. The one modality-specific layer
is the **serving front end**, which in v6 today is **vLLM only** (`run.sh` launches vLLM's OpenAI
server; the `vllm_metadata.json` / `EXTRA_MODELS_DIR` registration is LLM-specific).

Supporting diffusion / other model types is a **future extension** — a `serve.kind` on the manifest
that `render_run_sh` dispatches on (e.g. `vllm` today; a `custom` kind where `run.sh` launches
`model.py`'s own server). That needs a new serving contract + a `model.py` runner protocol + a
different API surface, so it is **deliberately out of scope for this draft** (tracked in #29) — not a
drop-in. The shape it would take:

```mermaid
flowchart TB
  subgraph shared1["SHARED — same v6 bundle for every model type"]
    w["Pointer to weights"] --- man["Manifest (serve.kind + config)"] --- mp["model.py"] --- venv["uv venv: ttnn + tt-metal-models + generic_op"]
  end
  shared1 --> fork{{"manifest: serve.kind ?"}}
  fork -->|"vllm  (implemented)"| llm
  fork -->|"diffusion | custom  (planned)"| other
  subgraph llm["LLM"]
    l1["run.sh → vLLM OpenAI server"] --> l2["vllm_metadata.json (EXTRA_MODELS_DIR)"] --> l3["model.py = generator adapter"] --> l4["POST /v1/chat/completions"]
  end
  subgraph other["Diffusion / other — NOT in this draft"]
    o1["run.sh → model.py's own server"] --> o2["model.py brings the API"] --> o3["model.py = pipeline (UNet/VAE)"] --> o4["POST /v1/images/generations"]
  end
  llm --> eng["SHARED ENGINE — ttnn + generic_op on the TT card (+ SFPI, firmware)"]
  other --> eng
```

The **diffusion model code** itself would live in `tt-metal-models`, not tt-model — the engine
already runs it; only the serving layer is the gap.

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
| the **exact dep versions** you ran with — `tt-metal-models==…` (pulls the matching `ttnn`) | `requirements.txt` pins |
| the **`vllm-tt-plugin`** wheel (the vLLM integration) | `wheels/`, installed by path |
| the **vLLM** you served with — stock upstream, built `VLLM_TARGET_DEVICE=empty` | `install.sh` step (`deps.vllm`); optionally a prebuilt wheel via `--vllm-wheel` |
| *(if you wrote custom ops)* a **`generic_op` wheel** you built | `wheels/`, installed by path |
| the **serving entrypoint** — the HF architecture name + the `module:Class` the vLLM plugin loads | `vllm_metadata.json` (`arch` → `main_class`) |
| the **weights** (an HF repo id) + the **serving knobs** you validated (mesh, max_num_seqs, block_size) | manifest pointer + `resources`/`mesh` |

The key discipline: **pin what actually worked** — `pip freeze` in your working venv gives the real
`tt-metal-models` (and, until it lands, `ttnn`) version to put in `requirements.txt`.

### Steps
1. **Make `model.py` importable by its class path.** Class `QwenForCausalLM` in `model.py` → the
   entrypoint is `model:QwenForCausalLM` (module = filename without `.py`; at serve time `PYTHONPATH`
   is the bundle root).
2. **Write `requirements.txt`** with the versions your box ran — **ttnn / tt-metal-models only**:
   ```
   # tt-metal-models==<X>       # the models tree (incl. tt_transformers); pins ttnn==<X> exactly
   #                            # (upstream tt-metal#54340) — pulls the matching ttnn transitively
   ttnn==0.77.0                 # engine (PyPI today; pin directly until tt-metal-models lands)
   ```
   **Do NOT put `vllm` in `requirements.txt`.** vLLM is installed by `install.sh` as stock upstream
   vLLM built `VLLM_TARGET_DEVICE=empty` (the [`vllm-tt-plugin`](https://github.com/tenstorrent/vllm-tt-plugin)
   recipe) — a resolvable `vllm` pin would pull the CUDA wheel and clobber that build. SFPI + firmware
   are external box deps — also **not** in `requirements.txt`. (Omit `--requirements` and `package-thin`
   writes this as a template with the `tt-metal-models` TODO pin.)
3. **Ship the `vllm-tt-plugin` wheel** with `--plugin-wheel`. vLLM core builds from source at install
   by default; to avoid that (hermetic, faster), build a wheel once with
   `VLLM_TARGET_DEVICE=empty pip wheel --no-deps vllm==0.25.1` and pass it via `--vllm-wheel`.
4. **(Custom ops only)** `model.py` calls `ttnn.generic_op(...)`; ship your built wheel with `--ops-wheel`.
5. **Run `package-thin`:**
   ```bash
   tt-model package-thin <org>/<model> \
     --model-py ./model.py \
     --requirements ./requirements.txt \      # ttnn/tt-metal-models pins (or omit for the #29 template)
     --plugin-wheel ./wheels/vllm_tt_plugin-*.whl \   # vllm-tt-plugin (the vLLM integration)
     --vllm-wheel ./wheels/vllm-0.25.1-*.whl \  # optional: prebuilt empty-target vLLM (else built at install)
     --ops-wheel ./wheels/my_model_ops-*.whl \  # optional: your generic_op wheel (repeatable)
     --arch blackhole \
     --arch-name QwenForCausalLM --main-class model:QwenForCausalLM \
     --weights Qwen/Qwen3-4B \                # pointer, never embedded
     --mesh P150 --max-num-seqs 32 --block-size 64 \
     --out ./bundle                           # stage locally (omit + pass <org>/<model> to push)
   ```
6. **Result** — the bundle layout shown above (`model.py` + `requirements.txt` + `vllm-overrides.txt`
   + `wheels/` [vllm-tt-plugin + ops (+ optional vLLM wheel)] + `vllm_models/<name>/vllm_metadata.json`
   + manifest + `install.sh`/`run.sh`; no `metal/`, no embedded ttnn wheel, no vLLM fork).
7. **Round-trip** — on any card + firmware + SFPI box: `tt-model pull <org>/<model>` builds the venv
   (ttnn/models + empty-target vLLM + plugin) and fetches the weights; `tt-model serve <org>/<model>`
   launches it.

**Required from you:** `model.py` + a `requirements.txt` of real pins + *(optional)* a `generic_op`
wheel + the entrypoint (`--arch-name`/`--main-class`) + a weights repo id. Everything else is generated.

## Testing it in the lab (today)
1. `tt-model package-thin ... --out /tmp/thin`
2. Edit `/tmp/thin/requirements.txt` — pin the real `ttnn` now, and `tt-metal-models` once tt-metal#54340 publishes.
3. `bash /tmp/thin/install.sh` — builds the venv (ttnn → empty-target vLLM → plugin). Needs SFPI on
   the box; the default vLLM step builds from source (needs a C/C++ toolchain + network for
   `common.txt`). Pass `--vllm-wheel` at package time to install a prebuilt empty-target wheel instead.
4. `bash /tmp/thin/run.sh` (serves), then `curl` the endpoint — or `tt-model pull`/`serve` from HF.
