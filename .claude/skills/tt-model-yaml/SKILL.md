---
name: tt-model-yaml
description: Author a tt-model.yaml — the CONTAINER (v5.1) manifest only — for a model in a tt-metal checkout, by reading its serve recipe and import closure and interviewing the user for what the directory cannot tell you. Use for "containerize this model", "write a tt-model.yaml", "package this as a Docker image", or "make a v5.1/container package". Do NOT use for a v5 "fat" bundle (`tt-model package --from-metal --ttnn-wheel …`) or a v6 "thin" bundle (`tt-model package-thin …`) — those are authored with CLI flags and have no manifest file; this skill would produce a YAML they cannot read.
---

# Author a tt-model.yaml from a model directory

You are given a model directory inside a tt-metal checkout (e.g.
`models/demos/blackhole/<x>` or `models/autoports/<x>`). Write the one YAML file that
`tt-model package --container` turns into a self-contained Docker image.

**The manifest records what was VALIDATED, not what is current.** Everything below follows
from that: pins come from the environment that actually served the model, the allowlist
comes from what the serve path actually loads, and anything you cannot verify you ASK
about — never guess.

Reference: `examples/container-example.yaml` in this repo is the annotated template and
lists every field. Read it first.

## Step 0 — Confirm this is a CONTAINER package

tt-model has three packaging paths and **only one uses a YAML manifest**:

| package | authored with | ships |
| --- | --- | --- |
| **v5.1 container** ← this skill | `tt-model.yaml` + `package --container <yaml>` | an OCI image; consumer needs Docker + a TT card |
| v5 "fat" | CLI flags: `package <repo> --from-metal … --ttnn-wheel …` | wheels + a metal tree; consumer builds a venv |
| v6 "thin" | CLI flags: `package-thin …` | pip dependency pins; consumer builds a venv |

There is no manifest file for v5 or v6 — a `tt-model.yaml` handed to them is ignored, and
`package` without `--container` never reads one. So before writing anything, make sure a
container is what they want. Signals it is:

- they said "container", "Docker image", "v5.1", or "consumer needs nothing installed"
- they want a consumer to run the model **without** tt-metal, vLLM or a venv on the host

If instead they want the venv-based bundles, stop and point them at
`tt-model package --help` / `tt-model package-thin --help`. If it is genuinely unclear,
ask — the two produce different artifacts for different consumers, and the choice is not
reversible without a rebuild.

## Step 1 — Assume nothing else is installed

The user may have **no tt-inference-server, no model_spec.json, no release config** — just
their checkout and a way they run the model. Do not look for a settings database. The
values it would have held (concurrency, context, mesh, tt config) come from the launch
recipe if one exists, and from the user otherwise (Step 6).

## Step 2 — Gather evidence from the model directory

Read, skipping what does not exist:

1. **The launch recipe** — `serve*.sh`, `run*.sh`, `setup*.sh` in the model dir, the repo
   root, or a `quickstart/` nearby. Highest-value input; most fields transcribe from it.
   Ask the user where they last served from if you cannot find it.
2. `README.md` in the model dir — validated hardware, context length, quirks.
3. `vllm_bundle/` or `vllm_ext/` — a per-model registration folder. The directory whose
   **children** hold `vllm_metadata.json` is `runtime.extra_models_dir`. Read the metadata:
   its `hf_model` is usually the weights id, and `main_class` tells you the adapter.
4. `requirements.txt`, `overrides.txt` — pins and the comments explaining them.
5. **The working venv**, if there is one. Ground truth for pins — when it disagrees with
   the README, the venv wins:
   ```bash
   <venv>/bin/python -c "import importlib.metadata as md; print(md.version('vllm'))"
   <venv>/bin/python -c "import importlib.metadata as md; print(md.distribution('vllm-tt-plugin').read_text('direct_url.json'))"
   ```
   A `direct_url.json` naming a local path means they run an editable checkout — that path
   is what `runtime.plugin.path` should be.

## Step 3 — Choose the kind

| Evidence | `kind` |
| --- | --- |
| stock `vllm==X.Y.Z` from PyPI + a separate `vllm-tt-plugin`; launch is `vllm serve …` | `vllm-plugin` |
| the `tenstorrent/vllm` **fork** (plugin in-tree); launch is `python -m models.common.readiness_check.run_vllm_server …` | `vllm-fork` |

No local `tenstorrent/vllm` clone and a standalone plugin checkout ⇒ `vllm-plugin`.
`vllm-fork` additionally needs `runtime.model_dir` (the launcher's `--model-dir`), covered
by `source.code`.

## Step 4 — Build the `source.code` allowlist (the part agents get wrong)

The allowlist names EXACTLY what ships; the image contains **no other** `models/` code, so
a miss fails at build — or worse, at a consumer's first weight-load. Compute the closure:

```bash
grep -rhoE "(from|import) +models\.[a-zA-Z0-9_.]+" <model_dir> models/common | sort -u
```

Iterate: for each new top-level package it surfaces, grep *that* too, until closed. Almost
every model needs `models/common`.

**Then hunt for runtime DATA** — code reading files relative to the model dir:

```bash
grep -rnE "Path\(__file__\)|_MODEL_DIR|\.json[\"']|\.yaml[\"']" <model_dir>/tt <model_dir>/*.py
```

A config loaded with a **silent fallback** is load-bearing numerics. A real model read its
precision policy and context contract from `config/`, and `_supported_context()` swallowed
`OSError` — omit that directory and it serves at the wrong precision with no error
anywhere. Ship it, and add a `verify:` assertion (Step 7).

Two failure modes to avoid:
- **Over-listing**: bring-up debris (`readiness_*` results, `.refpt` tensors, `generated/`,
  large `doc/` trees) must not ship. Prefer subpaths when the directory is big.
- **Under-listing**: if the model's registration shim computes its root by directory depth
  (`Path(__file__).parents[N]`), the whole path from `models/` down must ship intact, or it
  resolves for the author and fails in the image.

If the model directory is small (a few MB), listing it whole is simpler and safer than
cherry-picking.

## Step 5 — Pin the runtime

- **`plugin`** — the default is the user's own checkout:
  `plugin: {path: /abs/path/to/vllm-tt-plugin}`. It is staged into the image like the
  tt-metal tree, so uncommitted work ships and nothing is fetched. Use
  `{repo, ref: <sha>}` only for CI or a sha someone else must fetch — and it **must be
  pushed**, or the build cannot clone it. Never a branch name: a plugin that moved under a
  validated model is the exact bug this path exists to prevent.
- **`vllm`** — `{version: "X.Y.Z"}` from the venv/script. If they built their own
  empty-target wheel, `{wheel: /path/dist/vllm-*.whl}` is faster and more faithful.
- **`extra_models_dir`** — the directory whose CHILDREN hold `vllm_metadata.json`
  (Step 2.3). Must be covered by `source.code`.
- **`lock`** — omit on the first build; `package` writes `requirements.lock` out, and the
  user commits it and sets `lock: requirements.lock` for reproducible rebuilds.

## Step 6 — Interview for what the directory cannot tell you

Ask about anything you did not find evidence for. Ask in ONE batch, with your best guess
and where it came from, so the user is confirming rather than composing:

| Field | Ask | Why it cannot be guessed |
| --- | --- | --- |
| `repo` | HF org/name to publish to | not in the checkout |
| `weights` | HF id; pin a `revision`? | a bare id follows the default branch — the consumer may get different weights |
| `hardware` + `mesh_device` | which board, e.g. `p300x2` / `P300x2` | the plugin's closed enum; chip counts must agree |
| `max_num_seqs` | concurrent users: 1 (interactive) or N (fleet) | **required** — the TT backend rejects vLLM's default |
| `block_size` | paged-attention block size | **required**, same reason |
| `max_model_len` | served context | often LOWER than the model advertises |
| `additional_config.tt` | `trace_region_size`, `fabric_config`, `sample_on_device_mode` | model-specific tuning |
| `capabilities.tool_parser` | tool-calling model? which parser | otherwise tool calls come back as prose |
| `ubuntu` / `python` | base image + interpreter | must match a published tt-metalium dev image tag |

If a launch script exists, transcribe from it and ask only to confirm. If the README states
a context or hardware target, quote it in the question.

`serve_profiles:` is OPTIONAL — omit it for a single configuration and put everything in a
flat `serve:` block. Add profiles only when ONE image should serve several configs the user
actually validated (different device targets, or latency vs capacity); `default_profile` is
then required.

## Step 7 — `verify:` assertions

One Python statement per entry, run inside the finished image. Cover what nothing else
catches:

- every **silent-fallback data file** from Step 4:
  `"from pathlib import Path; p = Path('/opt/tt-metal/<rel>'); assert p.is_file(), '<what breaks>'"`
- the adapter import: `"from models.<...> import <Class>; assert <Class>"`
- any lazy import the closure surfaced.

Do NOT assert the registration shim imports directly — it is importable only once something
has put its folder on `sys.path`. tt-model already resolves every model registered through
`EXTRA_MODELS_DIR` for you.

## Step 8 — Validate without building

Write `<model_dir>/tt-model.yaml`, then:

```bash
python -c "
from tt_kernel.container_manifest import load_container_manifest
m = load_container_manifest('<model_dir>/tt-model.yaml', check_sources=True)
p = m.resolve_profile()
print('VALID:', m.name, m.kind, '|', p.hardware, p.mesh_device, 'seqs', p.max_num_seqs, 'block', p.block_size)
"
```

`check_sources=True` proves every `source.code` path exists. This validates shape, the
mesh/hardware cross-check, the kind-specific `runtime:` block, and that
`extra_models_dir` is covered by the allowlist.

**`tt-model serve <the yaml>` will not work** — serve takes the BUILT manifest
(`tt_kernel_manifest.json`) or a published `org/name`. The authored YAML describes how to
build; there is no image yet.

To preview the launch command before committing to a build:

```bash
python -c "
from tt_kernel.container_manifest import load_container_manifest
from tt_kernel.launchers import launcher_for
m = load_container_manifest('<model_dir>/tt-model.yaml')
w = m.to_wire(image_tag='preview', tt_metal_version='x', tt_kernel_version='0')
L = launcher_for(m.kind); p = w.container.resolve_profile()
print(' '.join(L.serve_argv(w, p))); print(L.serve_env(w, p))
"
```

Diff that against the working serve script flag-for-flag. A missing flag here is a broken
deployment later.

## Finish

Report: the chosen `kind` and why; the allowlist with one line of justification per entry;
every pin and where it came from; and anything you could NOT verify as an explicit open
question, not a silent guess. Then offer the next step:

```bash
tt-model package --container <model_dir>/tt-model.yaml --out ~/tt-model-builds
```

Warn that a cold build takes minutes to hours, and that the first serve JIT-compiles
kernels (~10 min) before the server reports ready.
