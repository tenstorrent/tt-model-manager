<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# End-to-end recipe (v6): package, push, pull, serve

This is the copy-paste recipe for the **v6 thin** path. Take a model you have brought up, package
it into one self-contained bundle, publish it to the Hugging Face (HF) Hub, and let anyone with a
Tenstorrent PCIe card pull and serve it. Nothing outside the install folder is needed to run it.

> Scope: **v6 thin bundles only.** These build their own engine and venv from pip pins and bundled
> wheels. A consumer needs only a Tenstorrent card, its firmware, and SFPI. For the v5.1 container
> path, which ships the platform as an OCI image instead, see [E2E_RECIPE_V5.1.md](E2E_RECIPE_V5.1.md).

> **Agents:** follow the [`tt-model-thin-package`](../.claude/skills/tt-model-thin-package/SKILL.md)
> skill. It is this recipe as a checklist, with the verification steps that make "it works" a
> claim you can defend.

There are two roles:

- **Producer**: the host where the model already serves.
- **Consumer**: any host with a Tenstorrent card, firmware, and SFPI. Nothing else is required.

---

## The canonical shape

A v6 bundle is four layers. Each has one right answer:

```
<bundle>/                      schema 6, `deps` block
  model.py                     entry point: imports your ops wheel, re-exports the vLLM adapter class
  extra_code/<model_pkg>/      YOUR model's code; run.sh puts extra_code/ first on PYTHONPATH
  requirements.txt             ttnn==<version on PyPI> (+ torchvision). No index options in this file.
  constraints.txt              fully resolved pins for every install step
  wheels/
    <model>_ops-*.whl          custom kernels via ttnn.generic_op — only if you need any
    tt_transformers-*.whl      TOOLING ONLY, no models/ (until tt-transformers publishes to PyPI)
    vllm_tt_plugin-*.whl
    vllm-*+empty-*.whl         prebuilt empty-target vLLM
  vllm_models/<name>/vllm_metadata.json
  install.sh  run.sh  tt_kernel_manifest.json
# weights: NOT here. A pointer in the manifest.
```

| layer | canonical | do NOT |
|---|---|---|
| engine | **stock `ttnn` from PyPI** | ship a forked or custom-built `ttnn` (a `+local` version tag is the tell) |
| custom kernels | a **`generic_op` wheel** (`--ops-wheel`) | compile kernels into `ttnn` |
| tooling | **tt-transformers, tooling only** | contribute your model *into* tt-transformers |
| model code | **inside the bundle** (`--extra-code`) | pin a whole-`models/`-tree wheel (such as `tt-metal-models`) to run one model |
| dependency safety | **`--constraints`** with fully resolved pins | rely on index order, or put `--extra-index-url` in `requirements.txt` |
| capabilities | **`--tool-parser` / `--reasoning-parser`** | leave them unset |

**Why the model ships in the bundle.** tt-transformers is becoming the tooling needed to run models,
with the models themselves living elsewhere. Pinning a wheel of the whole `models/` tree to run one
model ships every other model's code with yours (thousands of files to run a few dozen) and
couples your bundle to all of it. Your model depends on tt-transformers' **tooling API** only —
in practice `modules.lightweightmodule`, `sampling`, `tensor_utils`, `device_utils`, and
`modules.tt_ccl` — and travels with its own bundle.

**Why a `generic_op` wheel, not a ttnn fork.** The engine is then the stock, published `ttnn`, and
your custom surface is a small, separately versioned wheel that `deps.custom_ops` records. A fork
hides the same kernels inside a ~90 MB `ttnn` and declares no custom surface at all.

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
- Your model's code as an importable package, serving correctly on this host.
- Any custom kernels built as a `generic_op` wheel, running on **stock** `ttnn`.
- The tt-transformers you developed against (to cut a tooling-only wheel from).
- The `vllm_tt_plugin` wheel and a prebuilt empty-target vLLM wheel.
- A `constraints.txt` with every resolved package pinned (`uv pip freeze` from your working venv).

---

## Step 0: Bring up your model (producer)

Bring the model up as **its own package**, against stock `ttnn` and tt-transformers' tooling.
- **Stock HF architecture** (Llama, Qwen, Mistral, and others): you may need no model code at all.
- **New architecture**: implement it in your own package — not inside tt-transformers. If you
  wrote custom kernels, build them as a `generic_op` wheel that your code calls through
  `ttnn.generic_op(...)`, not as a modified `ttnn`.

When the model produces coherent text on your host, you are ready to package. Record the exact
serving configuration that worked (context, sequences, block size, the `tt` additional config,
env, tool and reasoning parsers): the bundle must reproduce it, not approximate it.

---

## Step 1: Lay out the bundle inputs (producer)

1. **Tooling-only tt-transformers wheel.** From the wheel you developed against, delete everything
   in `tt_transformers/models/` except `__init__.py`, remove any `Requires-Dist` on your own
   packages, drop `RECORD`, and repack with `python -m wheel pack`.
2. **`extra_code/<model_pkg>/`.** Your model package. If it lived under `tt_transformers.models`,
   move it and rewrite its absolute self-imports; then confirm there are no dynamic or string
   imports, that data files moved with it, and that no `__file__`-relative path changed meaning.
   **Prove every import resolves** against `extra_code/` plus the tooling wheel before going on.
3. **`model.py`.** Import your ops wheel explicitly, then re-export the adapter class:
   ```python
   import <model>_ops  # noqa: F401  (deps.custom_ops is recorded from model.py's imports)
   from <model_pkg>.<adapter_module> import <AdapterClass>
   ```
4. **`requirements.txt` and `constraints.txt`.** `ttnn==<version>` (+ `torchvision`) with **no index
   options**, and a fully resolved constraints file.
5. **Install guards.** `--verify` statements that fail the install, not the first request: the
   ttnn version, a tooling-only tt-transformers, and your model code present under `extra_code/`.
   `install.sh` runs them with the venv's python and does not export `$HERE`; locate the bundle
   from `sys.prefix` (the venv is `<bundle>/venv`).

---

## Step 2: Package, staged locally (producer)

Stage first. **Omit the positional `<org>/<model-name>`** — with it, `package-thin` stages *and*
pushes.

```bash
tt-model package-thin --out ./bundle --name <bundle-name> \
  --model-py model.py --extra-code extra_code \
  --requirements requirements.txt --constraints constraints.txt --python 3.12.14 \
  --plugin-wheel wheels/vllm_tt_plugin-*.whl \
  --ops-wheel    wheels/<model>_ops-*.whl \
  --ops-wheel    wheels/tt_transformers-*.whl \
  --vllm-wheel   wheels/vllm-*+empty-*.whl --vllm-version <v> --vllm-common-requirements vllm-common.txt \
  --arch blackhole --arch-name <HFArchitecture> --main-class model:<AdapterClass> \
  --weights <org>/<weights> --weights-revision <sha> \
  --mesh <topology> --device-count <n> --mesh-fabric <fabric> \
  --max-model-len <n> --max-num-seqs <n> --block-size <n> --trace-region-bytes <n> \
  --tt-config-json '<json>' --tool-parser <name> --reasoning-parser <name> \
  --env KEY=VALUE --extra-arg <token> \
  --verify '<python statement>' --provenance provenance.json
```

| Flag | Meaning |
|---|---|
| `--model-py <file>` | the entry point; `--main-class model:<AdapterClass>` resolves against it |
| `--extra-code <dir>` | your model's code; its top-level packages go on PYTHONPATH ahead of the runner |
| `--requirements` / `--constraints` | index pins / a fully resolved pin set applied to every install step. **Constraints are required** (see Troubleshooting) |
| `--plugin-wheel <whl>` | the `vllm_tt_plugin` wheel, installed by path |
| `--ops-wheel <whl>` | *(repeatable)* wheels installed by path after vLLM: your `generic_op` wheel, and — today — the tooling-only tt-transformers wheel |
| `--vllm-wheel` | a prebuilt empty-target vLLM wheel, for a hermetic install |
| `--arch` | the card's ISA: `blackhole` or `wormhole_b0`. The one **fatal** compatibility gate |
| `--arch-name` / `--main-class` | the HF architecture and adapter class, written to `vllm_metadata.json` |
| `--weights <hf-id>` / `--weights-revision` | **pointer** to the weights, pinned. The weights are not embedded |
| `--tool-parser` / `--reasoning-parser` | serving capabilities; tool calling is unavailable without the first |
| `--tt-config-json` / `--extra-arg` / `--env` | the validated serving configuration, verbatim |
| `--verify '<stmt>'` | *(repeatable)* a `python -c` check run at the end of `install.sh` |
| `--provenance <file>` | **path** to a JSON file staged as `PROVENANCE.json` |
| `--out <dir>` | where to stage. Without the positional repo id, nothing is pushed |

Keep the full command in a script so the bundle can be rebuilt exactly. Then **inspect**:
- the vLLM argv — expand `run.sh`'s `CMD=(…)` through bash; each JSON argument must be **one** token
- the env — everything your validated configuration needs
- `deps.custom_ops` in the manifest — exactly your ops wheel(s), never the tooling wheel
- `run.sh`'s PYTHONPATH — `extra_code/` first

---

## Step 2b: Prove it locally (producer, needs a card)

`tt-model serve` takes a pulled `<org>/<name>`, not a local v6 folder, so prove the staged bundle
with its own scripts — on a **copy**, so the folder you push stays clean:

```bash
cp -r ./bundle /tmp/bundle-test
bash /tmp/bundle-test/install.sh          # exit 0, "All installed packages are compatible", guards pass
bash /tmp/bundle-test/run.sh --port 20000 # wait for "Application startup complete"
```

Before the multi-minute boot, an **import smoke test** catches import errors in seconds: with
`run.sh`'s `LD_PRELOAD` and `PYTHONPATH=/tmp/bundle-test/extra_code:/tmp/bundle-test`, run
`import model` and confirm the adapter loads from `extra_code/` and nothing loads from the model's
old location.

Then run [Step 6](#step-6-verify-consumer)'s checks against it. If a known-good build of the model
exists (a v5.1 container, an earlier bundle), compare a `temperature: 0` completion: identical text
is the strongest evidence the packaging changed nothing.

---

## Step 3: Push (producer)

Before pushing, look at the target repo. If it exists, check which bundle it holds
(`tt_kernel_manifest.json` `name`) and its visibility — pushing replaces it.

Push by re-running the Step 2 command **with** the positional `<org>/<model-name>` and the same
flags (`tt-model push` is for v5.1 container packages). A new repo is created **private**.
Visibility is tri-state and **a push does not flip it implicitly**; but `--private` or `--public`
passed against an existing repo **changes** it, so pass one only on purpose. See
[publishing.md](publishing.md#push-vs-public-vs-publish). Large wheels go to Git Large File
Storage (LFS) automatically.

Confirm the upload is what you tested: stage the push into a separate `--out` and diff it against
the folder from Step 2 — only the manifest's `created_at` should differ.

---

## Step 4: Pull (consumer)

On any host with a card, firmware, and SFPI:

```bash
tt-model pull <org>/<model-name>                # add --with-weights to pre-download the weights
```

This materializes the folder, then runs its `install.sh`, which, **entirely inside the folder**,
provisions the pinned Python interpreter (via `uv`, into `.python/`), builds the venv, installs
the pinned deps and the bundled wheels, and runs the bundle's `--verify` guards. Weights are
fetched from the HF pointer — with `--with-weights` now, otherwise on first serve — into
`<install>/.hf` by default, or `$HF_HOME` (`$HF_HUB_CACHE`) if you set it for both commands.

If the bundle's `arch` does not match this host's card, `pull` stops here with a clear message.

---

## Step 5: Serve (consumer)

```bash
tt-model serve <org>/<model-name>                    # launches the OpenAI-compatible server
tt-model serve <org>/<model-name> --port 8001        # serve's own option; other args pass through to vLLM
tt-model serve <org>/<model-name> --print            # print the fully-resolved command + env, don't run
tt-model serve <org>/<model-name> --no-update-check  # skip the "is there a newer revision?" check
```

See [cli.md](cli.md#run-a-model) for the pass-through rule.

`serve` runs the bundle's `run.sh` in the bundle's **own venv**, pointing every cache and home
(`HF_HOME`, `TT_CACHE_PATH`, and others) **inside the folder**. The first serve downloads weights
(unless pre-fetched), builds the tensor cache, and JIT-compiles kernels — often 10 minutes or more
on a cold cache — before it logs **`Application startup complete`**. Later starts are much faster.

`serve` also install-then-serves a not-yet-pulled bundle, so on a fresh host you can skip straight to
`tt-model serve <org>/<model-name>`.

### Updating to a newer published revision

When an installed bundle is served, `serve` makes one short (3 s bounded) Hub check and, if the
author has published a newer revision, prints a non-blocking advisory. To update, re-pull:

```bash
tt-model pull <org>/<model-name>      # reinstalls a stale bundle in place (an up-to-date one is reused)
```

A plain `pull` is the update path. You do **not** need `--force`, which is only for reinstalling
regardless, or overriding a compat or wheel warning, and which also skips those safety gates. If an
update's rebuild fails, the working install is kept. Disable the check per-serve with
`--no-update-check`, or entirely offline with `--local-only`.

---

## Step 6: Verify (consumer)

Run each check that applies:

1. **Identity** — `curl -s localhost:20000/v1/models` returns the manifest's weights id.
2. **Generation** — a deterministic completion returns coherent text:
   ```bash
   curl -s localhost:20000/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model":"<weights-id>",
          "messages":[{"role":"user","content":"Say hello in one sentence."}],
          "max_tokens":1500, "temperature":0}'
   ```
   For a **reasoning** model give it room: a small `max_tokens` returns `content: null` because the
   budget went to the `reasoning` field.
3. **Tool calling** — required when the bundle declares a tool parser: a request with `tools`
   returns `finish_reason: tool_calls` and arguments that are valid JSON.
4. **Reasoning** — with a reasoning parser, `reasoning` is populated separately from `content`.

Coherent text and working capabilities mean the full producer-to-consumer loop works. (`model` is
the weights repo id from the manifest, which `run.sh` also exports as `HF_MODEL`.)

---

## The self-containment guarantee (why this is safe to hand around)

After `pull`, a v6 install is **hermetic**: everything needed to serve lives under the install
folder: the interpreter, the venv, the engine, the model code, and, on first serve, the weights
and all caches. Serving depends on nothing outside the folder **except the Tenstorrent device,
system libc, and SFPI**. The only steps that touch the network are `pull` (to fetch the
interpreter and the pinned deps) and the weights download.

---

## Troubleshooting (the non-obvious parts)

| Symptom | Cause / fix |
|---|---|
| `pull`: `No solution found … certifi was found on https://download.pytorch.org/whl/cpu, but not at the requested version` | The requirements step ranks the PyTorch CPU index above PyPI, and uv takes any package that index carries — `certifi`, `urllib3`, `requests`, `idna` — only from it, at one old version. Your constraints file caught it. Install the offending wheel by path with `--ops-wheel` (after vLLM, no PyTorch index) instead of `--models-wheel`. |
| An install **succeeded** but the venv has `certifi 2022.12.7`, `requests 2.28.1`, `urllib3 1.26.13` | The same ranking, **without** a constraints file: a silent downgrade to a stale CA bundle and to releases predating published CVE fixes. Always package with `--constraints`, and never put `--extra-index-url` in `requirements.txt`. |
| `deps.custom_ops` is empty or lists the tooling wheel | It is recorded from `model.py`'s own imports. Import your ops wheel in `model.py`, and do not import tt-transformers there. |
| `tt-model serve ./bundle`: "not a tt-model bundle" | `serve` takes a pulled `<org>/<name>`; it does not serve a local v6 folder. Test locally with the bundle's own `install.sh` and `run.sh` (Step 2b). |
| `pull` refuses: arch mismatch | The bundle targets a specific ISA. Pull on a matching host, or repackage. |
| serve: "Failed to infer device type" | ttnn failed to import (static thread-local storage (TLS)). `run.sh` preloads `_ttnncpp.so` from `ttnn.libs/`. |
| serve: `Timed out while waiting for active ethernet core` | The devices are dirty, not the bundle: something was hard-killed earlier. Confirm nothing else holds them, `tt-smi -r`, and serve again. |
| serve: "Address already in use" | Another server holds the port. Use `tt-model serve <id> --port 8001`. |
| Tool calling not working | The bundle must declare a tool parser; `run.sh` emits `--enable-auto-tool-choice --tool-call-parser <name>`. |
| Nothing registers in vLLM | `vllm_metadata.json` must live in `vllm_models/<name>/` rather than the bundle root (the plugin scans children). |

See [docs/thin_packages.md](thin_packages.md) for the design details and the `deps` schema.
