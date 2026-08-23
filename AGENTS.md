# AGENTS.md — agent guide for tt-model-manager

This file is for AI agents (Claude Code, Codex, etc.) working in this repo or asked to
package models with it. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## What this tool is

A model is a **Docker image published on Hugging Face**. Three jobs, nothing else:
`package` (one YAML manifest → self-contained image + staged HF repo dir), `push`,
and `pull`/`serve`/`stop`/`logs`/`list`. Weights are the only thing that ever touches
the host — in the host's own HF cache, bind-mounted in. No host tt-metal, no venv
provisioning, no doctor.

Read `README.md` first; `docs/packaging.md` for the authoring walkthrough;
`docs/model_types.md` for the `vllm` / `vllm-legacy` type registry.

## Skills

**Creating a tt-model.yaml for a model in a tt-metal fork:** follow
[`.claude/skills/tt-model-yaml/SKILL.md`](.claude/skills/tt-model-yaml/SKILL.md).
Claude Code loads it as the `tt-model-yaml` skill; other agents (Codex included)
should open that file and execute it as a checklist. It encodes the paid-for lessons —
import-closure allowlists (lazy imports included), silent-fallback data files,
pinning the plugin sha from the validated venv instead of `main`.

## Design invariants — do not break these

1. **One YAML manifest is the whole authoring interface.** `tt-model package` takes
   exactly one argument. Never add per-field CLI flags that duplicate manifest fields.
2. **The manifest records what was VALIDATED.** `package` resolves every git ref to a
   sha and writes the `built:` provenance block. Never loosen a pin to a branch name in
   an example or a published manifest.
3. **Weights are a pointer, never embedded.** They live in the host HF cache; the
   container bind-mounts it read-write and never downloads at build time.
4. **One image serves all of a manifest's `serve_profiles`.** Hardware-varying config
   (`hardware`, `mesh_device`, `max_model_len`, `additional_config`) lives per-profile;
   only `arch` is baked. Never silently substitute a profile on hardware mismatch —
   warn, suggest, require `--force`.
5. **Model types are the extension point** (`src/tt_model/types/`). Type-specific work
   reaches the Dockerfile only through generated `install_engine.sh`/`verify.sh`; the
   Dockerfile itself stays type-agnostic. New engine → new type module + a row in
   `docs/model_types.md`, no schema or CLI changes.
6. **`stop` is SIGTERM-first** (`docker stop --timeout`, never `rm -f`); the entrypoint
   `exec`s so the server is PID 1. A hard kill dirties the mesh; reset only then.
7. **tt-model owns no plugin schema.** `vllm_metadata.json` belongs to the vLLM plugin;
   we ship the model's `vllm_ext` directory verbatim and point `EXTRA_MODELS_DIR` at it.

## Serve-path facts (regressions here are silent and expensive)

- `EXTRA_MODELS_DIR` must point at the PARENT of per-model folders, or 0 architectures
  register. `HF_MODEL` must be exported (adapters read env, not vLLM's `--model`).
  `VLLM_PLUGINS` must stay UNSET (it is an allow-list; setting it kills the model's
  tool/reasoning parsers).
- The hugepages mount is verbatim (`src=/dev/hugepages-1G,dst=/dev/hugepages-1G`) — umd
  regex-matches /proc/mounts.
- The runtime image needs the sfpi cross-compiler's own host libs
  (libmpc3/libmpfr6/libgmp10/libzstd1) and a tt-owned metal tree (python in it mkdirs
  on import). Both were found on hardware; both have comments at their fix sites.

## Workflow

- Every change lands via a PR to `main`; open PRs as draft. Never push to `main`.
- `pytest` must stay green (130 offline tests — no hardware, no docker daemon needed).
  The golden-string tests in `tests/test_serve_argv.py` pin validated launch recipes:
  if one fails, a recipe drifted — find out which side is wrong before "fixing" it.
- Long operations (`package`, `push`) stream live and survive Ctrl-C via the interrupt
  guard in `build.py`; keep that behavior for anything new that runs > 1 minute.
