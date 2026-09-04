---
name: tt-model-package-test
description: Package a model as a CONTAINER (v5.1) image from its tt-model.yaml and prove it works — build, serve on hardware, verify the API (tool calling included), stop cleanly, and optionally push. Use for "package this model", "build and test the container", "prove the package serves", "ship this model to HF". Requires an authored tt-model.yaml (use the tt-model-yaml skill to write one) and a box with the target TT devices. Do NOT use for v5 fat / v6 thin bundles — see AGENTS.md "Driving the flow" for those.
---

# Package a container model and prove it works

Input: a `tt-model.yaml` (written with the `tt-model-yaml` skill; `examples/container-example.yaml`
is the annotated reference). Output: a built, hardware-verified package — and a report that says
exactly what was verified and what was not. **Never claim the package works without the
hardware checkpoints below.**

In this repo the CLI runs as `uv run tt-model …` (from the repo root); an installed
`tt-model` works the same. With a venv already active, `uv run` prints `warning:
VIRTUAL_ENV=… does not match the project environment path .venv and will be ignored` —
that is correct behaviour (uv uses the project `.venv`), not a problem to fix. A cold build is 2.5–4 hours; a warm rebuild that only changes
the verify stage is minutes. Everything in Preflight exists to avoid burning the hours.

## Step 1 — Preflight (no docker, seconds)

Run all of these before the first build:

1. **Validate the manifest offline** and confirm every allowlist path exists:
   ```bash
   uv run python -c "
   from tt_kernel.container_manifest import load_container_manifest
   m = load_container_manifest('<yaml>', check_sources=True)
   p = m.resolve_profile()
   print('VALID:', m.name, m.kind, '|', p.hardware, p.mesh_device)"
   ```
2. **Preview the launch argv** (`launcher_for(m.kind).serve_argv(...)` — see the
   tt-model-yaml skill, Step 8) and diff it flag-for-flag against the serve script or
   command the model was actually validated with. A missing flag here is a broken
   deployment later.

   Many models ship no `serve*.sh`. In descending order of authority, diff against: a
   launch command in the model's `README.md`; the flags recorded in a benchmark or
   bring-up note; or, failing both, walk the argv WITH the user field by field and have
   them confirm it. Never skip the step for want of a script — say which source you
   used, so the report shows what the argv was actually checked against.
3. **If the manifest has `runtime.lock`, dry-run the resolution locally** with the same
   flags the image uses — a 2-second failure here is the same failure 40 minutes into
   the build:
   ```bash
   uv venv --python <source.python> /tmp/lockcheck -q
   uv pip install --python /tmp/lockcheck/bin/python --dry-run -r <lockfile> \
     --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
   ```
4. **Check the box**: the target devices exist and are idle (`tt-smi`, and no running
   `tt-model-*` container in `docker ps`), and — only if a push is planned —
   `hf auth whoami` shows an account that can write the manifest's `repo:`.

## Step 2 — Package

```bash
uv run tt-model package --container <yaml> --out ~/tt-model-builds
```

Run it in the background and give the user the live log up front:
`tail -f ~/.cache/tt-model/build/<name>.log`. On a TTY the first Ctrl-C only warns; a
second within the window cancels and keeps the caches, so a re-run is cheap.

The in-image verify stage runs LAST, after every layer is cached — a verify failure
re-runs in minutes, so iterate there without fear. Failure patterns seen on real models:

- **`ModuleNotFoundError` for a package nothing declares** (e.g. `pytest` imported at
  module level by `models/common/utility_functions.py`, reached via
  `tt_transformers.model_config`). A live resolve never installs it. Fix: seed
  `runtime.lock` from the validated venv (`uv pip freeze --python <venv>/bin/python`),
  then strip what the build supplies (editable installs, `file://` wheels, `vllm==*+*`)
  and anything broken or useless in the image (mutually inconsistent dev cruft — pip
  tolerates a broken venv, uv refuses to reproduce one — and CUDA-only `nvidia-*`/
  `cuda-*`/`flashinfer-*` stacks). Re-pin local wheels to their PyPI releases.
- **A `models/...` file missing from the image** at verify: the allowlist under-lists.
  Beware lazy imports (`from_pretrained` loading a `tests/` helper mid-boot) and
  data files read with a silent fallback — ship them and assert them in `verify:`.
- **Circular `current_platform` ImportError in a verify line**: import `vllm` FIRST in
  any check that touches `vllm_tt_plugin.platform` — the plugin activates during
  vLLM's own import, and a plugin-first import re-enters a half-initialized module.

After success, confirm the staged output exists:
`~/tt-model-builds/<name>/` with `tt_kernel_manifest.json`, `README.md`, `code/`,
`image/`, `requirements.lock`.

## Step 3 — Serve on hardware

`serve` and `stop` take the BUILT manifest (or a pulled `org/name`) — never the
authoring YAML:

```bash
uv run tt-model serve ~/tt-model-builds/<name>/tt_kernel_manifest.json [--profile <p>]
```

The first start loads weights and JIT-compiles kernels — minutes (some models ~4, some
~10). Ready = the kind's ready line in the log (`Application startup complete` for the
vLLM kinds). Do not declare failure early; do not declare success before it.

While it boots, verify the argv is the one you previewed: `docker ps` for the container,
then `ps -o args= -C vllm` inside or `pgrep -af "vllm serve"` on the host.

**If the boot dies on the mesh open** — the log carries `TT_THROW: Device <n>: Timed out
while waiting for active ethernet core <x-y> to become active again` — the devices are
dirty, not the package. Something was hard-killed earlier (a container that exited
non-zero, a SIGKILLed fabric server) and left eth cores untrained. Confirm nothing else
holds them (`docker ps -a --filter name=tt-model-`, `pgrep -af vllm`), then `tt-smi -r all`
and serve again; it cost one wasted boot on a real run.

Reset ONLY on that evidence. It reinitialises every board on the host, so a speculative
reset before each serve would disrupt anyone else on the box and add minutes to every
attempt for nothing. Preflight cannot pre-empt this: `tt-smi -ls` proves the boards
enumerate, not that their eth cores are trained, and the `ETH_LIVE_STATUS` word in
`tt-smi -s` is a per-arch bitmask — reading it wrong means resetting always or never
noticing.

## Step 4 — Prove it works

Run every check that applies; each one catches a real, observed failure mode.

1. **Identity** — the served model is the manifest's weights id:
   ```bash
   curl -s localhost:<port>/v1/models
   ```
2. **Generation** — a deterministic completion returns coherent text:
   ```bash
   curl -s localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "<weights-id>",
     "messages": [{"role": "user", "content": "Merge two sorted lists in Python."}],
     "max_tokens": 256, "temperature": 0}'
   ```
3. **Tool calling** — REQUIRED when the manifest declares `capabilities.tool_parser`
   (skip only when it declares none). Send a real `tools` request and check the verdict,
   not the prose:
   ```bash
   curl -s localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "<weights-id>",
     "messages": [{"role": "user", "content": "What is the weather in Paris right now, in Celsius?"}],
     "tools": [{"type": "function", "function": {"name": "get_weather",
       "description": "Get the current weather for a city.",
       "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}, "metric": {"type": "boolean"}},
         "required": ["city"]}}}],
     "tool_choice": "auto", "max_tokens": 256, "temperature": 0
   }' | python3 -c 'import sys, json; c = json.load(sys.stdin)["choices"][0]; \
   print(c["finish_reason"], json.dumps(c["message"].get("tool_calls")))'
   ```
   Pass: `finish_reason` is `tool_calls` and the arguments are valid JSON matching the
   schema. Fail: prose in `content` with `finish_reason` `stop` — the parser is not in
   the launch (check the argv for `--enable-auto-tool-choice --tool-call-parser <name>`).
4. **Reasoning** — when `capabilities.reasoning_parser` is set, the response carries a
   populated `reasoning` field separate from `content`.
5. **Every profile this box can run** — repeat 1–3 per profile with
   `--profile <name>` (stop between profiles). Report profiles the box CANNOT run as
   not verified; never silently substitute hardware.

## Step 5 — Stop cleanly

```bash
uv run tt-model stop ~/tt-model-builds/<name>/tt_kernel_manifest.json
```

SIGTERM-first; if docker had to SIGKILL, tt-model resets the mesh itself and says so.
If anything was killed outside tt-model, run `tt-smi -r all` before the next serve — a
hard-killed fabric server leaves eth cores dirty and the next mesh open fails.

## Step 6 — Push (only when the user asks)

```bash
uv run tt-model push ~/tt-model-builds/<name> [--private|--public] [--publish]
```

A new repo is created private by default. `--public` on an existing repo CHANGES its
visibility, and `--publish` lists it in the community catalog — both are outward-facing;
never add them on your own judgment. After the push, open the repo and check the
rendered card: description and hardware up front, correct quickstart, provenance links
that actually resolve.

## Report

End with two lists, and be exact: **verified** (each Step 4 check that ran, on which
hardware and profile) and **not verified** (everything else — other profiles, other
boards, un-exercised capabilities). "It built" is not "it works"; only Step 4 earns
that sentence.
