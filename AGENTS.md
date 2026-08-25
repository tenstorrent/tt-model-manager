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

1. **tt-model is the standalone path.** The full flow — `package → push → pull → install →
   serve` — MUST work with tt-model alone (only a TT card + firmware). Never add a dependency
   on `tt-cli`; tt-cli is an *optional* wrapper that calls tt-model, not the reverse.
   Host provisioning is `tt-model install` and lives in `provision.py`. It is the ONLY
   module that installs the surrounding platform: `doctor`, `toolchain`, `instances`, and
   the compatibility gates stay strictly declarative — they discover, probe, and report.
   That separation is what makes a `doctor` verdict trustworthy, so keep it.
2. **Distribution is HuggingFace, not GitHub Releases.** Bundles are HF `model` repos; large
   binaries go to git-LFS via `hub.upload_folder`. Do not add Release-based or ad-hoc download flows.
3. **Weights are a pointer, never embedded** (`WeightsRef` = HF repo id). Do not stage weights
   into a bundle.
4. **Self-contained ("fat") packages ship the author's built artifacts** — their `ttnn` wheel
   (custom kernels compiled in), base vLLM, plugin — plus their modified metal tree. The engine
   is what's on the box, not a stock pin.
5. **Manifest back-compat.** `manifest.py` must keep reading every version in
   `SUPPORTED_SCHEMAS` (v3/v4/v5). Add fields as optional; bump `SCHEMA_VERSION` only for a new
   authored schema and keep the old readers + a round-trip test.
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
`hub.py` (HF push/pull), `runtime.py` (`download_weights`, `pip_install_wheels`,
`install_self_contained`), `bundles.py` (EXTRA_MODELS_DIR materialization + metadata render),
`packaging.py` (staging), `provision.py` (host setup). Prefer extending these over new
parallel code paths.

**All terminal output goes through `console.py`.** No new `typer.secho`, no scattered
`print`. See [docs/cli_output.md](docs/cli_output.md) for the vocabulary and the rules that
are easy to get wrong — chiefly: capture subprocess noise and surface it only on failure,
never gate a failure on `show_detail()`, and keep `--print`/`--json` on `console.raw()` so
Rich cannot wrap a pasteable command or a JSON document.

**Scripts in `scripts/` are shims.** `install.sh` and `make_test_cache.sh` exist for the
bootstrap case only and forward to the CLI. Logic added there cannot reuse tt-model's own
detection and drifts from the CLI silently; `tests/test_install_script.py` enforces this.

## Don't
- Don't vendor `torch`/`vllm`/`transformers` — they are pip deps.
- Don't commit wheels or other large binaries to git (LFS on push only).
- Don't push to `main` or self-merge.
- Don't silently truncate or skip integrity checks / version gates.
