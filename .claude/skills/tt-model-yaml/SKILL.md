---
name: tt-model-yaml
description: Author a tt-model.yaml manifest for a model living in a tt-metal fork, by reading its working serve recipe and import closure. Use when someone says "package this model", "write a tt-model.yaml", "create a manifest for models/autoports/<x>", or points at a tt-metal model directory to containerize. Produces a manifest that `tt-model package` can build and validates it offline before finishing.
---

# Author a tt-model.yaml from a model directory

You are given a model directory inside a tt-metal fork (e.g.
`models/autoports/poolside_laguna_xs_2_1` or `models/demos/<x>`). Your job is to write
the one YAML file that `tt-model package` turns into a self-contained Docker image.

**The manifest records what was VALIDATED, not what is current.** Every judgement below
follows from that: pins come from the environment that actually served the model, the
code allowlist comes from what the serve path actually loads, and anything you cannot
verify gets surfaced to the user instead of guessed.

Reference examples (in tt-model-manager): `examples/laguna-xs-2.1.yaml` (type `vllm`),
`examples/ornith-1.0-35b.yaml` (type `vllm-legacy`). Field semantics:
`docs/packaging.md`, `docs/model_types.md`.

## Step 1 — Gather the evidence

Read, in this order (skip what doesn't exist):

1. `<model_dir>/serve_vllm.sh`, `setup_vllm.sh`, `run*.sh` — the launch recipe and env
   pins. This is the highest-value input; most fields transcribe from here.
   **The recipe may live OUTSIDE the model dir**: also check the repo root, a
   `quickstart/` or handoff directory near the checkout, and ask the user where the
   model was last served from.
2. `<model_dir>/README.md` — validated hardware, mesh, context length, quirks.
3. `<model_dir>/vllm_ext/` — if present: an installable vLLM extension
   (`runtime.extension`) with `extra_models/*/vllm_metadata.json`. **If absent**, the
   model is likely in the plugin's BUILTIN registry (grep the plugin checkout's
   `platform.py` for the model name): omit `runtime.extension`, and the plugin pin
   must be a commit/release that CONTAINS the registration — add a `verify:` entry
   asserting it (e.g. the class name appears in `inspect.getsource` of
   `vllm_tt_plugin.platform`).
4. `<model_dir>/requirements.txt`, `overrides.txt` — version pins and their comments.
5. **The working venv**, if one exists on this box (ask the user which venv last served
   the model). It is the ground truth for pins — **when it contradicts the README,
   the venv wins** (a README saying `pip install vllm-tt-plugin` while the venv runs
   an editable git checkout means the PyPI release does NOT yet work for this model):
   ```bash
   <venv>/bin/python -c "import importlib.metadata as md; d=md.distribution('vllm'); print(d.version)"
   <venv>/bin/python -c "import importlib.metadata as md; print(md.distribution('vllm-tt-plugin').read_text('direct_url.json'))"
   ```

## Step 2 — Choose the type

| Evidence | `type` |
| --- | --- |
| setup installs `vllm==X.Y.Z` (sdist, `VLLM_TARGET_DEVICE=empty`) + clones `tenstorrent/vllm-tt-plugin`; launch is `vllm serve …` | `vllm` |
| setup clones the `tenstorrent/vllm` **fork** (plugin in-tree at `plugins/vllm-tt-plugin`); launch is `python -m models.common.readiness_check.run_vllm_server --stages serve …` | `vllm-legacy` |

`vllm-legacy` additionally needs `runtime.model_dir: <model_dir>` (the launcher's
`--model-dir`), and its `runtime.vllm` is `{repo, ref}`, with no `runtime.plugin`.

## Step 3 — Build the `source.code` allowlist (the part agents get wrong)

The allowlist names EXACTLY what ships; the image contains **no other** `models/` code,
so a miss fails at build — or worse, at a consumer's weight-load. Compute the closure,
don't eyeball it:

```bash
# every models.* import reachable from the serve path — INCLUDING LAZY ONES inside
# functions (a real model lazily imported its own tests/ dir at weight-load time)
grep -rhoE "(from|import) +models\.[a-zA-Z0-9_.]+" <model_dir> models/common 2>/dev/null | sort -u
```

Then iterate: for each new top-level package the grep surfaces (e.g. `models/common`),
grep *it* too, until closed. Almost every autoport needs `models/common`; closures
crossing into `models/tt_transformers` (via `model_config` imports) are common too.

**List subpaths, never the model dir root.** Real autoport dirs carry bring-up debris —
`readiness_*` results (tens of MB), `.refpt` reference tensors, `generated/`, and
hundreds of MB under `doc/` — that must not ship. Name `tt/`, `vllm_ext/`, `tests/`
(only if the closure proved it), and the specific data subdirs (e.g.
`doc/datatype_sweep`), not the parent.

**Also hunt for runtime DATA files** — code that reads files relative to the model dir:

```bash
grep -rnE "os.path.join\(.*_MODEL_DIR|Path\(__file__\)|\.json[\"']|\.yaml[\"']" <model_dir>/tt <model_dir>/*.py 2>/dev/null
```

A config the model loads with a *silent fallback* (e.g. a precision policy that quietly
defaults when its JSON is missing) is load-bearing numerics: ship its directory AND add
a `verify:` assertion for it (step 6). Never rely on "it's under doc/, it's just docs".

## Step 4 — Serve settings and profiles

Transcribe the working launch command. Shared values go under `serve:`; each named
`serve_profiles:` entry deep-merges over it. Two reasons to have several profiles, both
real: different **device targets** (p150x2 vs p150x4 — usually different
`max_model_len`, `trace_region_size`, `fabric_config`) and different **deployment
shapes** (`max_num_seqs: 1` latency vs `32` capacity). Only add profiles the author
actually validated; `default_profile` is required when there is more than one.

- `max_num_seqs` and `block_size` are REQUIRED on every merged profile (the TT backend
  rejects vLLM's defaults).
- `mesh_device` is the plugin's closed enum (`N150 P100 P150 P150x2 N300 P300 N150x4
  P150x4 T3K P150x8 P300x2 TG BH-Galaxy`) or a literal `"(rows, cols)"`; its chip count
  must match the `hardware` label (`p150x4` → 4).
- `--additional-config '{"tt": {...}}'` from the script → `additional_config.tt` as YAML.
- Remaining launch flags → `args:` (use `[--flag, value]` pairs for valued flags);
  model-specific `TT_*` exports → `env:`.

## Step 5 — Pin the runtime (never `main`)

- `vllm` type: `runtime.vllm.version` = the exact version from the script/venv.
  `runtime.plugin` is `{repo, ref}` with **the commit sha from the working venv's
  `direct_url.json`** (step 1.5) — or `{version: "X.Y.Z"}` for a PyPI release that
  already registers the model. A branch name is a drift bomb: a real model packaged
  against `plugin@main` hit a first-decode TT_FATAL because main had moved past the
  validated sha. If the venv runs an editable checkout, pin THAT checkout's HEAD sha
  and confirm it is pushed somewhere reachable (`git branch -r --contains HEAD`); if
  the checkout is dirty in files this model touches, STOP and tell the user to commit
  and push first — an unreproducible pin is not a pin.
- `vllm-legacy`: `runtime.vllm.ref` = the fork sha the README/venv names.
- If a working venv exists, seed the lock:
  ```bash
  uv pip freeze --python <venv>/bin/python > <model_dir>/requirements.lock
  ```
  and set `runtime.lock: requirements.lock`. Otherwise omit — the first `package`
  writes it back.

## Step 6 — `verify:` assertions

One Python statement per entry, run inside the finished image. Cover the failure modes
nothing else catches:

- every LAZY import found in step 3 (`"import models...."`),
- existence of each silent-fallback data file
  (`"import pathlib; p = pathlib.Path('/opt/tt-metal/<rel>'); assert p.exists(), '...'"`),
- any env assertion the model's own setup script makes (e.g. a ttnn API the fork adds —
  for nanobind functions check `__doc__`, not `inspect.signature`).

## Step 7 — Assemble and validate

Top-level fields: `schema: 1`, `repo:` (ask the user for the HF org/name), `name:`,
`weights:` (the HF id the launch command serves), `type:`, `arch:`
(`blackhole`/`wormhole_b0` — from the validated hardware), `source:` (`tt_metal:` =
absolute path to this fork checkout; `code:`; `ubuntu: "22.04"`; `python:` = the
working venv's minor version).

Write the file as `<model_dir>/tt-model.yaml`, then validate WITHOUT building:

```bash
tt-model list  <model_dir>/tt-model.yaml            # profiles parse + hardware fit
tt-model serve <model_dir>/tt-model.yaml --print    # the exact docker run + launch argv
```

The `--print` output's launch command must match the working serve script's flags
one-for-one (order aside). Diff them; a missing flag here is a broken deployment later.
If `tt-model` is not installed: `uv tool install tt-model` (or validate with
`python -c "from tt_model.manifest import load_manifest; load_manifest('<file>')"`).

## Finish

Report to the user: the chosen type and why, the allowlist with one line of
justification per entry, every pin and where it came from, and anything you could NOT
verify (no working venv, unclear hardware target) as an explicit open question — not a
silent guess. Offer the next step: `tt-model package <model_dir>/tt-model.yaml`.
