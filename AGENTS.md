# AGENTS.md — guidelines for automated (agent) contributions to tt-model

This file tells an AI agent (e.g. Claude Code) how to make and submit improvement PRs to
`tt-model-manager` so fixes stay consistent with the design. Read it before changing
code. Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md); this is the agent-facing
supplement, and the invariants below are binding for everyone.

## Golden rule
**Every change lands via a PR. Never push to `main`, never merge your own PR.** Open PRs as
**draft** for human review; a human clicks merge.

## Design invariants — do not break these
tt-model exists to ship *self-contained* model packages over HuggingFace. A change that
violates one of these is wrong even if tests pass:

1. **tt-model is the standalone path.** The full flow — `package → pull → serve` — MUST work
   with tt-model alone; the box needs only a TT card + firmware (plus SFPI, an
   externally-managed box dep). Never add a dependency on `tt-cli`; tt-cli is an *optional*
   wrapper that calls tt-model, not the reverse. **There is no host provisioning.** Every
   bundle builds its OWN per-model venv (v5 from embedded wheels, v6 from pip pins), so
   tt-model never installs a shared platform and never relies on a pre-installed tt-metal/vLLM
   on the box. The compatibility check stays strictly declarative — it discovers the target
   arch/machine and reports a verdict; it does not provision.
2. **Distribution is HuggingFace, not GitHub Releases.** Bundles are HF `model` repos; large
   binaries go to git-LFS via `hub.upload_folder`. Do not add Release-based or ad-hoc download flows.
3. **Weights are a pointer, never embedded** (`WeightsRef` = HF repo id). Do not stage weights
   into a bundle.
4. **Two bundle schemas, both self-contained.** A **v5 "fat"** bundle (schema `5`, the
   `bundled` block, authored with `tt-model package`) ships the author's built artifacts —
   their `ttnn` wheel (custom kernels compiled in), an empty-target vLLM wheel, the plugin
   wheel — plus their modified `tt-metal-community` tree, installed into a fresh venv by
   `install.sh`. A **v6 "thin"** bundle (schema `6`, the `deps` block, authored with
   `tt-model package-thin`) builds the venv from pip pins (`ttnn` / `tt-metal-models`) plus
   bundled wheels (the `vllm-tt-plugin` + any `generic_op` wheel) plus an empty-target vLLM
   build step — no embedded `ttnn` wheel, no `metal/` tree. Either way the engine that serves
   is the one the bundle builds, never a shared box install.
5. **Manifest support is v5 + v6 only.** `manifest.py`'s `SUPPORTED_SCHEMAS` is
   `{"5", "6"}`; a bundle with any other `schema_version` is refused ("re-publish the bundle
   with a current tt-model"). Add fields as optional and keep a round-trip test; bump
   `SCHEMA_VERSION` only for a genuinely new authored schema.
6. **The serving contract is the plugin's `EXTRA_MODELS_DIR`**: per-model *subfolder* with
   `vllm_metadata.json` (`arch` + `main_class`). Keep `run.sh`/`stage_package` aligned with it.

## Serve-path facts the code encodes (regressions here are silent and expensive)
- The shipped `ttnn` wheel must bundle `_ttnncpp.so`; locate ttnn via
  `importlib.util.find_spec` (never `import ttnn`) when computing `LD_PRELOAD` — the import is
  what the preload fixes (glibc static-TLS).
- `run.sh` must `export HF_MODEL` (the tt_transformers adapter reads it from env, not vLLM
  `--model`) and emit `--max_num_seqs` + `--block_size` (the TT backend rejects vLLM's 256/None
  defaults).
- Single-chip runs disable fabric + set `TT_METAL_VISIBLE_DEVICES=0`.

## Workflow for a fix
1. Branch: `fix/<slug>` or `feat/<slug>` off `main`.
2. Keep the PR **small and single-concern**. One bug or one feature per PR.
3. Reproduce first, then fix. Add a **regression test** that fails before and passes after.
4. Run the full offline suite — it must stay green:
   ```
   pytest            # expected: all pass (no hardware, no network)
   ```
5. If the change touches the serve/device path, validate on hardware (see
   `docs/self_contained_packages.md` → Testing) — package → pull → serve → `curl`.
6. Commit messages: imperative subject, a short body explaining *why*, and end with:
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
7. Open a **draft PR**; body = what/why + the test evidence (e.g. "full suite: N passed") +
   a link to the tracking issue if one applies.

## Reuse, don't reinvent
`hub.py` (HF push/pull + catalog listing), `runtime.py` (`download_weights`,
`install_self_contained`), `packaging.py` (`stage_package`, `render_install_sh`,
`render_run_sh` — the running-folder layout + the EXTRA_MODELS_DIR / `vllm_metadata.json`
render), `manifest.py` (the v5/v6 schema + `compare()`, the compatibility verdict),
`metal.py`/`device.py` (arch/machine detection), `localdb.py` (installed-bundle bookkeeping).
Prefer extending these over new parallel code paths.

**All terminal output goes through `console.py`.** No new `typer.secho`, no scattered
`print`. See [docs/cli_output.md](docs/cli_output.md) for the vocabulary and the rules that
are easy to get wrong — chiefly: capture subprocess noise and surface it only on failure,
never gate a failure on `show_detail()`, and keep `--print`/`--json` on `console.raw()` so
Rich cannot wrap a pasteable command or a JSON document.

**The bundle's own `install.sh`/`run.sh` are generated, not hand-maintained.** They are
rendered from the manifest by `render_install_sh`/`render_run_sh` in `packaging.py`. Change the
rendered script there (with a test), never by editing a staged bundle — a bundle in the wild
carries whatever it was published with.

## Don't
- Don't vendor `torch`/`vllm`/`transformers` — they are pip deps.
- Don't commit wheels or other large binaries to git (LFS on push only).
- Don't push to `main` or self-merge.
- Don't silently truncate or skip integrity checks / version gates.
