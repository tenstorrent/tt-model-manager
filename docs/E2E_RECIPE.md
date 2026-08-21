<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# End-to-end recipe: model → package → push → pull → serve (v5)

This is the canonical, copy-paste recipe for the **v5 self-contained** path: take a model you've
brought up on `tt-metal-community`, package it into one self-contained bundle, publish it to the
Hugging Face Hub, and have anyone with a Tenstorrent card pull and serve it — with **nothing
outside the install folder** needed to run.

> Scope: **v5 self-contained bundles only.** These ship their own engine + venv; a consumer needs
> only a TT card + firmware. (The older v4 "kernels-less" and legacy dispatch paths are in the
> [README](../README.md); this recipe does not cover them.)

There are two roles:

- **Producer** — the box where the model already serves on `tt-metal-community`.
- **Consumer** — any box with a TT card + firmware. Nothing else is required.

---

## Prerequisites

**Both roles**
- Linux **x86_64**, Ubuntu **22.04 or 24.04**.
- A Tenstorrent card + firmware/driver (`/dev/tenstorrent/*` present).
- `tt-model` installed: `pip install tt-model-manager`.
- Authenticated to HF for push/pull of private repos:
  ```bash
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

**Producer only**
- A working `tt-metal-community` checkout that serves your model.
- Your **built `ttnn` wheel** (custom C++/LLK kernels compiled in) and the `vllm_tt_plugin` wheel,
  in one directory.
- `auditwheel` + `patchelf` (`pip install auditwheel patchelf`) — used to make the engine wheel
  portable.

---

## Step 0 — Bring up your model on tt-metal-community (producer)

Not covered in depth here — see `tt-metal-community`'s `CONTRIBUTING.md` / `docs/BRINGUP.md`.
Two cases:
- **Stock HF architecture** (Llama/Qwen/Mistral/…): no code — `./run_demo.sh <org>/<model-id>`.
- **New architecture**: implement the novel block against ttnn ops in
  `models/tt_transformers/tt/`; if you wrote custom C++/LLK kernels, rebuild *your* `ttnn` wheel so
  they're compiled in.

> **Cross-Ubuntu tip (build the engine wheel on 22.04):** the packaged `ttnn` wheel is tagged for
> the glibc of the box it's repaired on. Build/repair on **Ubuntu 22.04 (glibc 2.35)** and one
> bundle runs on **both 22.04 and 24.04**. Build on 24.04 (glibc 2.39) and it runs on 24.04 only —
> `pull` will refuse it on a 22.04 host with a clear glibc message.

When `./run_demo.sh <model>` produces coherent text, you're ready to package.

---

## Step 1 — Package (producer)

`tt-model package` snapshots your built artifacts into one bundle folder and (optionally) pushes it.
The HF target is a **positional** argument; omit it and pass `--out <dir>` to stage locally first.

```bash
tt-model package <your-org>/<model-name> \
  --from-metal /path/to/tt-metal-community \
  --wheels-dir /path/to/wheels \
  --arch blackhole \
  --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct \
  --mesh P150 \
  --vendor-deps \
  --repair
```

What the flags mean:

| Flag | Meaning |
|---|---|
| `--from-metal <dir>` | your modified tt-metal-community tree (embedded as `metal/`) |
| `--wheels-dir <dir>` | auto-classifies `ttnn-*` / `vllm-*` / `vllm_tt_plugin-*` (or pass `--ttnn-wheel`/`--plugin-wheel` explicitly) |
| `--arch` | the card ISA (`blackhole` / `wormhole_b0`) — the one **fatal** compatibility gate |
| `--arch-name` / `--main-class` | the HF architecture + adapter class → `vllm_metadata.json` |
| `--weights <hf-id>` | **pointer** to the weights — never embedded |
| `--mesh` | device topology (e.g. `P150`, `1x4`) |
| `--vendor-deps` | vendor the full dependency closure so install is **offline + reproducible** (recommended) |
| `--repair` | run auditwheel so the engine wheel is portable (`$ORIGIN` RPATH, vendored libs) — default |
| `--manylinux <policy>` | *(optional)* assert a glibc floor, e.g. `manylinux_2_28_x86_64` (see the cross-Ubuntu tip) |
| `--out <dir>` | stage locally instead of / before pushing |

The result is one HF **model** repo — the "running folder":

```
wheels/            your ttnn (+vllm +plugin) wheels + the vendored dep closure   (git-LFS)
metal/             your modified tt-metal-community tree
vllm_models/<name>/vllm_metadata.json     the EXTRA_MODELS_DIR contract
install.sh  run.sh  requirements.txt
tt_kernel_manifest.json                   the v5 manifest
# weights: NOT here — a pointer in the manifest
```

---

## Step 2 — Push (producer)

If you passed a positional `<org>/<model-name>` to `package`, it already pushed. To push a locally
staged folder, or to control visibility:

```bash
# package + push in one go (default: private)
tt-model package <org>/<model-name> ... --public          # or omit for private

# stage first, inspect, then push
tt-model package ... --out ./bundle                        # no push
# (push the staged folder by re-running package with the positional id, or push from your flow)
```

Visibility is tri-state and **a push never flips it implicitly** — see the README's *Repo
visibility* table. Large wheels go to git-LFS automatically.

---

## Step 3 — Pull (consumer)

On any box with a card + firmware:

```bash
tt-model pull <org>/<model-name>
```

This materializes the folder, then runs its `install.sh`, which — **entirely inside the folder** —
provisions the pinned Python interpreter (via `uv`, into `.python/`), builds the venv, and installs
the shipped wheels + deps (offline, from `wheels/`, when they were vendored). Weights are fetched
from the HF pointer (add `--with-weights` to pre-download; otherwise they're fetched on first serve).

If the bundle's engine wheel needs a newer glibc than this host has, `pull` stops here with a clear
message (repackage on Ubuntu 22.04). If the interpreter/arch don't match, same.

---

## Step 4 — Serve (consumer)

```bash
tt-model serve <org>/<model-name>                 # launches the OpenAI-compatible server
tt-model serve <org>/<model-name> --port 8001     # any extra args pass through to vLLM
tt-model serve <org>/<model-name> --print         # print the fully-resolved command + env, don't run
```

`serve` runs the bundle's `run.sh` in the bundle's **own venv**, pointing every cache/home
(`HF_HOME`, `TT_CACHE_PATH`, …) **inside the folder**. First serve does a weight download + JIT
warmup (single-chip: several minutes) before it logs **`Application startup complete`**. If a newer
revision of the bundle has been published, `serve` prints a one-line advisory suggesting a re-pull.

`serve` also install-then-serves a not-yet-pulled bundle, so on a fresh box you can skip straight to
`tt-model serve <org>/<model-name>`.

---

## Step 5 — Verify (consumer)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Llama-3.2-3B-Instruct",
       "messages":[{"role":"user","content":"Say hello in one sentence."}],
       "max_tokens":64}'
```

Coherent text back = the full producer→consumer loop works. (`--model` is the weights repo id from
the manifest, which `run.sh` also exports as `HF_MODEL`.)

---

## The self-containment guarantee (why this is safe to hand around)

After `pull`, a v5 install is **hermetic**: everything needed to serve lives under the install
folder — the interpreter, the venv, the engine (with your kernels), the model code, and, on first
serve, the weights and all caches. Serving depends on nothing outside the folder **except the TT
device and system libc**. The only step that touches the network is `pull` (fetch the interpreter,
and — unless `--vendor-deps` — the pip deps).

---

## Troubleshooting (the non-obvious parts)

| Symptom | Cause / fix |
|---|---|
| `pull` refuses: "needs glibc >= 2.39, host has 2.35" | Engine wheel built on Ubuntu 24.04, host is 22.04. Repackage with the wheel built/repaired on 22.04. |
| `pull` refuses: interpreter/arch mismatch | The shipped wheels are cp312/linux_x86_64 for a specific ISA — pull on a matching host, or repackage. |
| serve: "Failed to infer device type" | ttnn failed to import (static-TLS). `run.sh` preloads `_ttnncpp.so` from `ttnn.libs/`; ensure the wheel was `--repair`ed. |
| serve: "Address already in use" | Another server holds the port — `tt-model serve <id> --port 8001`. |
| Tool calling not working | The bundle must declare `capabilities.tool_parser`; `run.sh` emits `--enable-auto-tool-choice --tool-call-parser <name>`. |
| Nothing registers in vLLM | `vllm_metadata.json` must live in `vllm_models/<name>/`, not the bundle root (the plugin scans children). |

See [docs/self_contained_packages.md](self_contained_packages.md) for the design details and the
offline test commands.
