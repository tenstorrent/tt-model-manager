# AGENTS.md — guidelines for automated (agent) work with tt-model

This file tells an AI agent how to work with `tt-model-manager`. It covers two jobs:

- **Changing the tool's code** — the design invariants, serve-path facts, and PR discipline
  a fix must respect (most of this file).
- **Driving the flow** — using `tt-model` to package, publish, pull, and serve a model on
  behalf of a user (see [Driving the flow (using tt-model)](#driving-the-flow-using-tt-model)).

Read it before changing code. Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md); this
is the agent-facing supplement, and the invariants below are binding for everyone.

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
5. **Manifest support is v5 + v5.1 + v6.** `manifest.py`'s `SUPPORTED_SCHEMAS` is
   `{"5", "5.1", "6"}`; a bundle with any other `schema_version` is refused ("re-publish the
   bundle with a current tt-model") rather than silently half-read. Bump `SCHEMA_VERSION` only
   for a genuinely new authored schema.

   **v5.1 is the CONTAINER schema** (the `container` block): the platform ships as an OCI image
   rather than as a venv, so the consumer needs only Docker + a TT card. It is a POINT release of
   v5 because it makes the same promise — a package needing no host tt-metal — by a stronger
   mechanism. That numbering is deliberate: it left the whole number free, which is why v6 "thin"
   could take it without a collision.

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

---

## Driving the flow (using tt-model)

This section is for an AI agent *using* `tt-model` for a user — taking a model brought up on
`tt-metal-community`, packaging it, publishing it, and serving it anywhere with just a TT card.
It is about **running the flow**, not changing the tool; to change the code, follow the rest of
this file (draft PR, one concern, regression test, never push to `main`).

There are **two authoring paths**, both self-contained (a consumer needs only a TT card +
firmware). Pick the one the user's box supports:

- **v5 "fat"** — `tt-model package …` embeds the author's built artifacts (their `ttnn` wheel,
  an empty-target vLLM wheel, the plugin wheel, their `tt-metal-community` tree). **Runnable
  today.** Design: [docs/self_contained_packages.md](docs/self_contained_packages.md); copy-paste
  recipe: [docs/E2E_RECIPE.md](docs/E2E_RECIPE.md).
- **v6 "thin"** — `tt-model package-thin …` builds the venv from pip pins (`ttnn` /
  `tt-metal-models`) plus bundled wheels (`vllm-tt-plugin` + any `generic_op`). **Not fully
  runnable yet** — gated on `tt-metal-models` publishing (tt-metal#54478) and `tt_transformers`
  being broken out; the generated `requirements.txt` ships with a TODO pin until then. Design:
  [docs/thin_packages.md](docs/thin_packages.md).

### The canonical sequence

```
# v5 (runnable today)
bring up (tt-metal-community)  →  package        →  pull  →  serve  →  curl
# v6 (thin — same flow, gated on tt-metal-models)
bring up (tt-metal-community)  →  package-thin   →  pull  →  serve  →  curl
```

1. **package** (producer, v5): `tt-model package <org>/<name> --from-metal <dir> --wheels-dir
   <dir> --arch <isa> --arch-name <HFArch> --main-class <module:Class> --weights <hf-id> --mesh
   <mesh> --vendor-deps --repair` — pushes to HF unless you pass `--out <dir>` to stage locally.
   - **package-thin** (producer, v6): `tt-model package-thin <org>/<name> --model-py model.py
     --requirements requirements.txt [--plugin-wheel …] [--ops-wheel …] --arch <isa>
     --arch-name <HFArch> --main-class <module:Class> --weights <hf-id> --mesh <mesh>` — omit
     `--requirements` and it writes the template with the `tt-metal-models` TODO pin.
2. **pull** (consumer): `tt-model pull <org>/<name>` (`--with-weights` to pre-download weights).
3. **serve** (consumer): `tt-model serve <org>/<name> [--port N] [--print] [--local-only]`.
4. **verify**: `curl .../v1/chat/completions` → coherent text.

There is **no `push`/`install`/`run`/`start` command** — `package`/`package-thin` push, `pull`
installs, `serve` install-then-serves. Everything after the id on `serve` passes through to vLLM.

### Invariants when helping a user

These are the *using*-side restatements of the binding design invariants above — do not violate
them in a suggested workflow. Read them in full at
[Design invariants](#design-invariants--do-not-break-these):

1. **tt-model alone must do the whole job** (invariant 1). The only consumer prereq is a TT card
   + firmware (plus SFPI). Never introduce a step needing tt-cli, a host tt-metal, or host vLLM.
2. **The folder is the wall.** After `pull`, serving uses nothing outside the install directory
   except the TT device + system libc — interpreter (`.python/`), venv, engine, and caches all
   live inside. Never point the model at a shared/system cache to "fix" something.
3. **Weights are a pointer** (invariant 3), never embedded — `--weights <hf-id>`.
4. **The engine is what the bundle builds** (invariant 4): v5 serves the author's `ttnn` wheel
   (kernels compiled in, made portable with `auditwheel --repair`); v6 builds it from the pinned
   deps. Never substitute a stock/pinned wheel into a v5 bundle.
5. **glibc floor is real (v5).** Build/repair the engine wheel on the **oldest** target (Ubuntu
   22.04, glibc 2.35) to serve both 22.04 and 24.04. A wheel repaired on 24.04 is 24.04-only.
6. **Serving contract = the plugin's `EXTRA_MODELS_DIR`** (invariant 6): `vllm_metadata.json` in
   a per-model *subfolder* (`vllm_models/<name>/`), not the bundle root.

### Verification checkpoints — do not claim success without them

- **After package (v5):** the bundle has `wheels/` (incl. a `manylinux_*` ttnn wheel), `metal/`,
  `vllm_models/<name>/vllm_metadata.json`, `install.sh`, `run.sh`, `tt_kernel_manifest.json`.
  For **package-thin (v6):** `model.py`, `requirements.txt`, the bundled `vllm-tt-plugin` (+ any
  `generic_op`) wheel, `vllm_models/<name>/vllm_metadata.json`, `install.sh`, `run.sh`, manifest.
- **After pull:** `install.sh` succeeded; `<install>/venv/bin/python` exists.
- **After serve:** the log reaches **`Application startup complete`** (model load + JIT warmup is
  minutes on a single chip — wait, don't declare failure early). For a container package,
  `tt-model serve` itself waits and shows the boot as a checklist ending in a ready card;
  `--detach` skips the wait.
- **Only then** run the `curl` and confirm the text is coherent. Report the real result; if a step
  failed, say so with the output.

### Gotchas the flow encodes (each is a real past failure)

The serve-path mechanics — `find_spec`/`LD_PRELOAD`, `run.sh` exporting `HF_MODEL` and emitting
`--max_num_seqs`/`--block_size`, single-chip fabric-off — are covered under
[Serve-path facts](#serve-path-facts-the-code-encodes-regressions-here-are-silent-and-expensive).
Beyond those:

- Do **not** set `VLLM_PLUGINS` — it's an allow-list that silently suppresses the model's
  tool/reasoning-parser plugins.
- Tool calling: the manifest must declare `capabilities.tool_parser`; the launcher emits
  `--enable-auto-tool-choice --tool-call-parser <name>` (vLLM normalizes `_`→`-`).
- `serve` passes unknown args through to vLLM (e.g. `--port 8001`); `--print` echoes the resolved
  command+env without launching; `--local-only` requires an installed bundle and never hits the Hub.
- **Updating:** `serve` warns (best-effort, 3s-bounded; `--no-update-check` to skip) when a newer
  revision exists. The update path is a plain `tt-model pull <id>` — it reinstalls a stale bundle
  in place. Do **not** tell users to `pull --force` to update: `--force` also skips the
  compat/wheel gates and is only for reinstalling regardless or overriding a warning.

### When something's wrong

Prefer a clear, actionable message over a silent workaround. If `pull` refuses (glibc /
interpreter / arch), the fix is to **repackage on the right OS**, not to force past the gate.
Consult the Troubleshooting table in [docs/E2E_RECIPE.md](docs/E2E_RECIPE.md) (v5). For v6, if the
`tt-metal-models` pin can't resolve yet, that's the expected gate — the path is not runnable until
tt-metal#54478 publishes; see [docs/thin_packages.md](docs/thin_packages.md).
