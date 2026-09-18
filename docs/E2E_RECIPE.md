<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# End-to-end recipe (v6): package, push, pull, serve

This is the copy-paste recipe for the **v6 thin** path. Take a model you have brought up on
`tt-metal-community`, package it into one self-contained bundle, publish it to the Hugging Face
(HF) Hub, and let anyone with a Tenstorrent PCIe card pull and serve it. Nothing outside the
install folder is needed to run it.

> Scope: **v6 thin bundles only.** These build their own engine and venv from pip pins. A
> consumer needs only a Tenstorrent card, its firmware, and SFPI.

> **This path is gated on an upstream publish.** It depends on the **models wheel**,
> `tt-metal-models`, which packages the whole `models/` tree (including `tt_transformers`) for pip
> and pins `ttnn` exactly ([`tenstorrent/tt-metal#54478`](https://github.com/tenstorrent/tt-metal/pull/54478)).
> Until `tt-metal#54478` merges and publishes, the generated `requirements.txt` pins `ttnn` directly
> (it is on PyPI) and carries a `tt-metal-models` TODO pin. Once the wheel publishes, a thin bundle
> pins that one dep and is runnable end-to-end.

There are two roles:

- **Producer**: the host where the model already serves on `tt-metal-community`.
- **Consumer**: any host with a Tenstorrent card, firmware, and SFPI. Nothing else is required.

---

## Prerequisites

**Both roles**
- Linux **x86_64**, Ubuntu **22.04 or 24.04**.
- A Tenstorrent card with firmware and driver installed (`/dev/tenstorrent/*` present).
- SFPI (the SFPU programming interface compiler), an externally managed host dependency. See
  [thin_packages.md](thin_packages.md#host-prerequisites).
- `tt-model` installed. It is **not on PyPI**; install it from a clone:
  ```bash
  git clone https://github.com/tenstorrent/tt-model-manager && cd tt-model-manager
  python -m venv .venv && source .venv/bin/activate
  pip install -e .
  ```
  There is nothing else to provision on the host. Each bundle builds its **own** per-model venv
  from what it pins, so any `ttnn` or vLLM on the host is not required and is not touched. See
  the [README](../README.md#install).
- Authenticated to HF for push and pull of private repos, either with `tt-model login` or:
  ```bash
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

**Producer only**
- A working `tt-metal-community` checkout that serves your model.
- Your `model.py` runner and a `requirements.txt` of the exact `ttnn` / `tt-metal-models` pins it
  ran with (`pip freeze` in your working venv gives them).
- The `vllm_tt_plugin` wheel, plus any `generic_op` custom-op wheel you built.

---

## Step 0: Bring up your model on `tt-metal-community` (producer)

Not covered in depth here. See `tt-metal-community`'s `CONTRIBUTING.md` and `docs/BRINGUP.md`.
Two cases:
- **Stock HF architecture** (Llama, Qwen, Mistral, and others): no code. Run
  `./run_demo.sh <org>/<model-id>`.
- **New architecture**: implement the novel block against ttnn ops in
  `models/tt_transformers/tt/`. If you wrote custom ops, build them as a `generic_op` wheel that
  `model.py` calls through `ttnn.generic_op(...)`.

When `./run_demo.sh <model>` produces coherent text, you are ready to package.

---

## Step 1: Package (producer)

`tt-model package-thin` snapshots your runner, pins, and wheels into one bundle folder and
(optionally) pushes it. The HF target is a **positional** argument; omit it and pass `--out <dir>`
to stage locally first.

```bash
tt-model package-thin <your-org>/<model-name> \
  --model-py ./model.py \
  --requirements ./requirements.txt \
  --plugin-wheel ./vllm_tt_plugin-*.whl \
  --ops-wheel ./generic_op-*.whl \
  --arch blackhole \
  --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct \
  --mesh P150
```

What the flags mean:

| Flag | Meaning |
|---|---|
| `--model-py <file>` | your runner; `--main-class` resolves against it at serve time |
| `--requirements <file>` | the `ttnn` / `tt-metal-models` pins. Omit it to have `package-thin` write a template `requirements.txt` |
| `--plugin-wheel <whl>` | the `vllm_tt_plugin` wheel, installed by path |
| `--ops-wheel <whl>` | *(optional, repeatable)* custom-op wheels |
| `--arch` | the card's instruction set architecture (ISA): `blackhole` or `wormhole_b0`. The one **fatal** compatibility gate |
| `--arch-name` / `--main-class` | the HF architecture and adapter class, written to `vllm_metadata.json` |
| `--weights <hf-id>` | **pointer** to the weights. The weights are not embedded |
| `--mesh` | device topology (for example `P150`, `1x4`) |
| `--out <dir>` | stage locally instead of, or before, pushing |

The result is one HF **model** repo, the "running folder":

```
model.py                                  your runner
requirements.txt                          the ttnn / tt-metal-models pins
wheels/            vllm-tt-plugin (+ any generic_op) wheels, installed by path   (git-LFS)
vllm_models/<name>/vllm_metadata.json     the EXTRA_MODELS_DIR contract
install.sh  run.sh
tt_kernel_manifest.json                   the v6 manifest
# weights: NOT here. A pointer in the manifest.
```

For the full bundle layout, the remaining authoring flags, and the offline tests, see
**[docs/thin_packages.md](thin_packages.md)**.

---

## Step 2: Push (producer)

If you passed a positional `<org>/<model-name>` to `package-thin`, it already pushed. To control
visibility, pass a flag on the same command:

```bash
# package + push in one go (default: private)
tt-model package-thin <org>/<model-name> ... --public     # or omit for private

# stage first, inspect, then push
tt-model package-thin ... --out ./bundle                   # no push
```

`tt-model push` does not accept a v6 staging directory (it is for v5.1 container packages). To
push after inspecting a staged folder, re-run `tt-model package-thin` with the positional
`<org>/<model-name>` and the same flags.

Visibility is tri-state and **a push does not flip it implicitly**. See
[publishing.md](publishing.md#push-vs-public-vs-publish). Large wheels go to Git Large File
Storage (LFS) automatically.

---

## Step 3: Pull (consumer)

On any host with a card, firmware, and SFPI:

```bash
tt-model pull <org>/<model-name>
```

This materializes the folder, then runs its `install.sh`, which, **entirely inside the folder**,
provisions the pinned Python interpreter (via `uv`, into `.python/`), builds the venv, installs
the pinned deps, builds the empty-target vLLM, and installs the bundled wheels. Weights are
fetched from the HF pointer. Add `--with-weights` to pre-download them; otherwise they are fetched
on first serve.

If the bundle's `arch` does not match this host's card, `pull` stops here with a clear message.

---

## Step 4: Serve (consumer)

```bash
tt-model serve <org>/<model-name>                    # launches the OpenAI-compatible server
tt-model serve <org>/<model-name> --port 8001        # serve's own option; other args pass through to vLLM
tt-model serve <org>/<model-name> --print            # print the fully-resolved command + env, don't run
tt-model serve <org>/<model-name> --no-update-check  # skip the "is there a newer revision?" check
```

See [cli.md](cli.md#run-a-model) for the pass-through rule (which options belong to `serve` and
which go to vLLM).

`serve` runs the bundle's `run.sh` in the bundle's **own venv**, pointing every cache and home
(`HF_HOME`, `TT_CACHE_PATH`, and others) **inside the folder**. First serve does a weight download
and just-in-time (JIT) kernel warmup (single-chip: several minutes) before it logs
**`Application startup complete`**.

`serve` also install-then-serves a not-yet-pulled bundle, so on a fresh host you can skip straight to
`tt-model serve <org>/<model-name>`.

### Updating to a newer published revision

When an installed bundle is served, `serve` makes one short (3 s bounded) Hub check and, if the
author has published a newer revision, prints a non-blocking advisory. To update, re-pull:

```bash
tt-model pull <org>/<model-name>      # reinstalls a stale bundle in place (an up-to-date one is reused)
```

A plain `pull` is the update path. You do **not** need `--force`, which is only for reinstalling
regardless, or overriding a compat or wheel warning, and which also skips those safety gates.
Disable the check per-serve with `--no-update-check`, or entirely offline with `--local-only`.

---

## Step 5: Verify (consumer)

```bash
curl -s http://localhost:20000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Llama-3.2-3B-Instruct",
       "messages":[{"role":"user","content":"Say hello in one sentence."}],
       "max_tokens":64}'
```

Coherent text back means the full producer-to-consumer loop works. (`--model` is the weights repo
id from the manifest, which `run.sh` also exports as `HF_MODEL`.)

---

## The self-containment guarantee (why this is safe to hand around)

After `pull`, a v6 install is **hermetic**: everything needed to serve lives under the install
folder: the interpreter, the venv, the engine, the model code, and, on first serve, the weights
and all caches. Serving depends on nothing outside the folder **except the Tenstorrent device,
system libc, and SFPI**. The only step that touches the network is `pull` (to fetch the
interpreter, the pinned deps, and vLLM's sources unless a prebuilt wheel was bundled).

---

## Troubleshooting (the non-obvious parts)

| Symptom | Cause / fix |
|---|---|
| `pull`: the `tt-metal-models` pin cannot resolve | The models wheel is not published yet (`tt-metal#54478`). Pin `ttnn` directly for now, or build the wheel yourself and ship it with `--models-wheel`. |
| `pull` refuses: arch mismatch | The bundle targets a specific ISA. Pull on a matching host, or repackage. |
| serve: "Failed to infer device type" | ttnn failed to import (static thread-local storage (TLS)). `run.sh` preloads `_ttnncpp.so` from `ttnn.libs/`. |
| serve: "Address already in use" | Another server holds the port. Use `tt-model serve <id> --port 8001`. |
| Tool calling not working | The bundle must declare `capabilities.tool_parser`; `run.sh` emits `--enable-auto-tool-choice --tool-call-parser <name>`. |
| Nothing registers in vLLM | `vllm_metadata.json` must live in `vllm_models/<name>/` rather than the bundle root (the plugin scans children). |

See [docs/thin_packages.md](thin_packages.md) for the design details, the bundle layout, and the
lab test steps.
