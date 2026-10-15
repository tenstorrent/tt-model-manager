# AGENTS.md: guidelines for automated (agent) work with `tt-model`

This file tells an AI agent how to work with `tt-model-manager`. It covers two jobs:

- **Changing the tool's code**: the design invariants, serve-path facts, and PR discipline
  a fix must respect (most of this file).
- **Driving the flow**: using `tt-model` to package, publish, pull, and serve a model on
  behalf of a user (see [Driving the flow](#driving-the-flow-using-tt-model)).

Read it before changing code. Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md). This
is the agent-facing supplement, and the invariants below are binding for everyone.

## Golden rule
**Every change lands via a PR. Do not push to `main`, and do not merge your own PR.** Open PRs as
**draft** for human review; a human clicks merge.

## Design invariants: do not break these
`tt-model` exists to ship *self-contained* model packages over Hugging Face. A change that
violates one of these is wrong even if tests pass:

1. **`tt-model` is the standalone path.** The full flow (`package`, `pull`, `serve`) MUST work
   with `tt-model` alone; the host needs only a Tenstorrent PCIe card and firmware (plus SFPI, an
   externally managed host dep). Do not add a dependency on `tt-cli`; `tt-cli` is an *optional*
   wrapper that calls `tt-model`; `tt-model` does not call `tt-cli`. **There is no host provisioning.** Every
   bundle builds its OWN per-model venv (v6 from pip pins), so `tt-model` does not install a
   shared platform and does not rely on a pre-installed TT-Metalium™ or vLLM on the host. The compatibility check stays strictly declarative: it
   discovers the target arch and machine and reports a verdict; it does not provision.
2. **Distribution is Hugging Face.** Bundles are HF `model` repos; large
   binaries go to Git Large File Storage (LFS) via `hub.upload_folder`. Do not add Release-based
   or ad-hoc download flows.
3. **Weights are a pointer and are not embedded** (`WeightsRef` = HF repo id). Do not stage
   weights into a bundle.
4. **Two bundle schemas, both self-contained.**
   - A **v5.1 container** bundle (the `container` block) ships the platform as an Open Container
     Initiative (OCI) image rather than a venv, so the consumer needs only Docker and a
     Tenstorrent card.
   - A **v6 "thin"** bundle (schema `6`, the `deps` block, authored with `tt-model package-thin`)
     builds the venv from pip pins (**stock** `ttnn`) plus bundled wheels, with no embedded or
     forked `ttnn` wheel and no `metal/` tree. tt-transformers is a **tooling** dependency only; the
     model's own code ships in the bundle (`deps.extra_code_dir`), and custom kernels ship only as
     `generic_op` wheels. See [docs/E2E_RECIPE_V6.md](docs/E2E_RECIPE_V6.md#the-canonical-shape). `deps.kind` picks the serving front end
     `render_run_sh` puts in `run.sh`: `"vllm"` (the default, and v6's only behavior before this
     field existed) adds an empty-target vLLM build step (the `vllm-tt-plugin` plus any
     `generic_op` wheel) and serves `vllm.entrypoints.openai.api_server`; `"tt-dit-server"` (no
     vLLM step regardless of `--vllm`/`--no-vllm`) serves `deps.app` directly with uvicorn
     instead, for a model with no tokens, KV cache, or continuous batching, mirroring the
     same-named v5.1 CONTAINER kind (`launchers.TtDitServerLauncher`). The model's code — and any
     hand-written **pure-Python** ops (composing stock `ttnn`, no new C++ op or forked build) —
     ships in `deps.extra_code_dir`, which rides in the bundle and goes on PYTHONPATH ahead of the
     runner; `deps.verify` holds install-time sanity checks (issue #127).

   In every case the engine that serves is the one the bundle builds rather than a shared host install.
   A v6 bundle's custom-op (`generic_op`) wheels are **auto-recorded in `deps.custom_ops`** from what
   `model.py` imports — metadata only (discovery/provenance), never an install list; `--ops-wheel`
   shipping and the by-path install are unchanged. Only `model.py`'s own imports are read, so a
   model whose code lives in `extra_code/` must import its ops wheel in `model.py` to be recorded.
5. **Manifest support is gated on `SUPPORTED_SCHEMAS`.** A bundle whose `schema_version` is not
   in `manifest.py`'s `SUPPORTED_SCHEMAS` is refused ("re-publish the bundle with a current
   `tt-model`") rather than silently half-read. Bump `SCHEMA_VERSION` only for a genuinely new
   authored schema.

## Serve-path facts the code encodes (regressions here are silent and expensive)
- The shipped `ttnn` wheel must bundle `_ttnncpp.so`. Locate ttnn via
  `importlib.util.find_spec` (not `import ttnn`) when computing `LD_PRELOAD`; the import is
  what the preload fixes (glibc static thread-local storage).
- `run.sh` must `export HF_MODEL` (the tt_transformers adapter reads it from env, not vLLM
  `--model`) and emit `--max_num_seqs` and `--block_size` (the Tenstorrent backend rejects
  vLLM's 256/None defaults).
- Single-chip runs disable fabric and set `TT_METAL_VISIBLE_DEVICES=0`.
- The serving contract is the plugin's `EXTRA_MODELS_DIR`: `vllm_metadata.json` in a per-model
  *subfolder* (`vllm_models/<name>/`) rather than the bundle root.

## Workflow for a fix
1. Branch: `fix/<slug>` or `feat/<slug>` off `main`.
2. Keep the PR **small and single-concern**. One bug or one feature per PR.
3. Reproduce first, then fix. Add a **regression test** that fails before and passes after.
4. Run the full offline suite. It must stay green:
   ```
   pytest            # expected: all pass (no hardware, no network)
   ```
5. If the change touches the serve/device path, validate on hardware (see
   `docs/E2E_RECIPE_V6.md` → Step 2b and the `tt-model-thin-package` skill for a v6 bundle, or the
   `tt-model-package-test` skill for a container package) — package → pull → serve → `curl`.
6. Commit messages: imperative subject and a body explaining *why* — the rationale that
   doesn't belong in a code comment goes here. **No AI attribution**: no `Co-Authored-By`
   trailer, no "Generated with Claude Code", no mention of Claude or Anthropic, in commits
   or PRs.
7. Open a **draft PR**; body = what/why + the test evidence (e.g. "full suite: N passed") +
   a link to the tracking issue if one applies. Keep it **brief, in plain, simple English**:
   a few short sentences and bullets, no jargon or walls of text.

## Code style
- **Comments are brief.** One line stating what isn't obvious from the code; no paragraphs,
  no restating the code, no history ("used to…", "now…"). The *why* in depth goes in the
  commit message.
- **Code is effective and clear, not verbose.** No speculative abstractions, redundant
  checks, or defensive branches for cases that can't happen. Match the surrounding code.
- Docstrings: one line unless the contract genuinely needs more.

## Reuse, don't reinvent
`hub.py` (HF push/pull and catalog listing), `runtime.py` (`download_weights`,
`install_self_contained`), `packaging.py` (`stage_package`, `render_install_sh`,
`render_run_sh`: the running-folder layout and the EXTRA_MODELS_DIR / `vllm_metadata.json`
render), `manifest.py` (the bundle manifest schema and `compare()`, the compatibility verdict),
`metal.py`/`device.py` (arch and machine detection), `localdb.py` (installed-bundle bookkeeping).
Prefer extending these over new parallel code paths.

**All terminal output goes through `console.py`.** No new `typer.secho`, no scattered
`print`. See [docs/cli_output.md](docs/cli_output.md) for the vocabulary and the rules that
are easy to get wrong. Chiefly: capture subprocess noise and surface it only on failure,
do not gate a failure on `show_detail()`, and keep `--print`/`--json` on `console.raw()` so
Rich cannot wrap a pasteable command or a JSON document.

**The bundle's own `install.sh`/`run.sh` are generated rather than hand-maintained.** They are
rendered from the manifest by `render_install_sh`/`render_run_sh` in `packaging.py`. Change the
rendered script there (with a test) rather than by editing a staged bundle. A bundle in the wild
carries whatever it was published with.

## Related tooling

[`tenstorrent/skills`](https://github.com/tenstorrent/skills) is a separate Claude Code and Codex
plugin marketplace for TT-Metalium bring-up, review, and debugging work (`tt-model-bringup`,
`tt-autodebug`, `tt-review-skills`, `tt-skills`). It does not overlap with or replace anything in
this repo: `tt-model` packages, publishes, and serves already-built models, while that marketplace
helps an agent bring a model up on TT-Metalium in the first place. This repo's own skills
(`tt-model-yaml`, `tt-model-package-test`, `cli-design`, and `feature-branch-pr`, under
`.claude/skills/` and `.codex/skills/`) are specific to authoring and testing `tt-model` packages
and to this repo's conventions, and are not published there. Install the marketplace with
`/plugin marketplace add git@github.com:tenstorrent/skills.git` then `/plugin install
tt-skills@tenstorrent-skills` (Claude Code) if bring-up work on the model you are packaging would
benefit from it.

## Don't
- Don't vendor `torch`/`vllm`/`transformers`; they are pip deps.
- Don't commit wheels or other large binaries to git (LFS on push only).
- Don't push to `main` or self-merge.
- Don't silently truncate or skip integrity checks or version gates.

---

## Driving the flow (using `tt-model`)

This section is for an AI agent *using* `tt-model` for a user: taking a model brought up on
`tt-metal-community`, packaging it, publishing it, and serving it anywhere with just a Tenstorrent
card. It is about running the flow rather than changing the tool. To change the code, follow the rest of
this file (draft PR, one concern, regression test, no pushes to `main`).

For the **container (v5.1)** path this repo ships two skills (under `.claude/skills/`, mirrored
for Codex under `.codex/skills/`): `tt-model-yaml` authors the manifest from a validated
bring-up; `tt-model-package-test` builds it, serves it on hardware, proves the API works (tool
calling included), and pushes. Prefer them over improvising the flow. The copy-paste recipe for
this path is [docs/E2E_RECIPE_V5.1.md](docs/E2E_RECIPE_V5.1.md).

There is one **venv-based authoring path**, also self-contained (a consumer needs only a card
and firmware):

- **v6 "thin"**: `tt-model package-thin …` builds the venv from pip pins (**stock** `ttnn`) plus
  bundled wheels. For this path the repo ships the `tt-model-thin-package` skill (same two
  locations); follow it rather than improvising. Copy-paste recipe:
  [docs/E2E_RECIPE_V6.md](docs/E2E_RECIPE_V6.md); design and `deps` schema:
  [docs/thin_packages.md](docs/thin_packages.md).

  The canonical v6 bundle has one right answer per layer — use it, and do not reproduce the
  alternatives that exist in published bundles:

  | layer | canonical | do NOT |
  |---|---|---|
  | engine | stock `ttnn` from PyPI | ship a forked or custom-built `ttnn` |
  | custom kernels | a `generic_op` wheel (`--ops-wheel`) | compile kernels into `ttnn` |
  | tooling | tt-transformers, **tooling only** | contribute the model *into* tt-transformers |
  | model code | inside the bundle (`--extra-code`) | pin a whole-`models/`-tree wheel (e.g. `tt-metal-models`) to run one model |
  | dependency safety | `--constraints`, fully resolved | rely on index order; index options in `requirements.txt` |
  | capabilities | `--tool-parser` / `--reasoning-parser` | leave them unset |

### The canonical sequence

1. **Bring up** the model as **its own package** against stock `ttnn` and tt-transformers'
   tooling; custom kernels as a `generic_op` wheel.
2. **Lay out** the inputs: a tooling-only tt-transformers wheel, the model in `extra_code/`
   (every import proven to resolve), a `model.py` that imports the ops wheel, `requirements.txt`
   with no index options, a fully resolved `constraints.txt`, and `--verify` guards.
3. **package-thin, staged** (producer). **Omit the positional `<org>/<name>`**, which pushes:
   ```
   tt-model package-thin --out <dir> --model-py model.py --extra-code extra_code \
     --requirements requirements.txt --constraints constraints.txt \
     --plugin-wheel … --ops-wheel <ops>.whl --ops-wheel <tooling>.whl --vllm-wheel … \
     --arch <isa> --arch-name <HFArch> --main-class model:<Class> --weights <hf-id> \
     --mesh <mesh> --tool-parser … --reasoning-parser … --verify '…'
   ```
4. **Prove it locally**: run the staged bundle's `install.sh` and `run.sh` on a copy, and verify.
5. **Push** (only when asked): re-run step 3 **with** `<org>/<name>`, after checking the target repo.
6. **pull** (consumer): `tt-model pull <org>/<name>` (`--with-weights` to pre-download weights).
7. **serve** (consumer): `tt-model serve <org>/<name> [--port N] [--print] [--local-only]`.
8. **verify**: identity, coherent text, tool calling and reasoning where declared.

There is no `install`, `run`, or `start` command. `package-thin` with a positional repo id stages
and pushes; without one it only stages. `pull` installs; `serve` installs then serves.
`tt-model push` exists only for a staged v5.1 container directory, and `serve` takes a pulled
`<org>/<name>`, not a local v6 folder. See the pass-through rule in [docs/cli.md](docs/cli.md#run-a-model) for which `serve`
options are its own and which go to vLLM.

### Invariants when helping a user

Do not violate the [Design invariants](#design-invariants-do-not-break-these) or the
[Serve-path facts](#serve-path-facts-the-code-encodes-regressions-here-are-silent-and-expensive)
above in a suggested workflow. In particular:

- The only consumer prerequisite is a Tenstorrent card and firmware (plus SFPI). Do not introduce
  a step needing `tt-cli`, a host TT-Metalium, or host vLLM.
- After `pull`, serving uses nothing outside the install directory except the Tenstorrent device
  and system libc: interpreter (`.python/`), venv, engine, and caches all live inside. Do not
  point the model at a shared or system cache to "fix" something.
- Weights are a pointer (`--weights <hf-id>`).
- The engine is what the bundle builds: v6 builds it from the pinned deps, and that engine is
  **stock** `ttnn`. Never ship a forked `ttnn` to carry custom kernels; ship a `generic_op` wheel.
- The model's code ships with its bundle. Do not add a model to tt-transformers to make a bundle
  work; it is tooling only.

### Verification checkpoints: do not claim success without them

- **After package-thin (v6):** the bundle has `model.py`, `extra_code/<model>/`,
  `requirements.txt`, `constraints.txt`, the bundled wheels, `vllm_models/<name>/vllm_metadata.json`,
  `install.sh`, `run.sh`, `tt_kernel_manifest.json`. `deps.custom_ops` lists exactly the
  `generic_op` wheel(s), never the tooling wheel; every JSON argument in `run.sh`'s argv expands to
  one token.
- **Before pushing (v6):** the staged bundle's `install.sh` passed on a copy (with its `--verify`
  guards), and `import model` loaded the adapter from `extra_code/`.
- **After pull:** `install.sh` succeeded; `<install>/venv/bin/python` exists.
- **After serve:** the log reaches **`Application startup complete`**. Model load plus JIT warmup
  takes minutes on a single chip; wait rather than declaring failure early. For a container
  package, `tt-model serve` itself waits and shows the boot as a checklist ending in a ready
  card; `--detach` skips the wait.
- **Only then** run the `curl` and confirm the text is coherent. Report the real result; if a step
  failed, say so with the output.

### Gotchas the flow encodes (each is a real past failure)

The serve-path mechanics (`find_spec`/`LD_PRELOAD`, `run.sh` exporting `HF_MODEL` and emitting
`--max_num_seqs`/`--block_size`, single-chip fabric-off) are covered under
[Serve-path facts](#serve-path-facts-the-code-encodes-regressions-here-are-silent-and-expensive).
Beyond those:

- Do **not** set `VLLM_PLUGINS`. It is an allow-list that silently suppresses the model's
  tool and reasoning-parser plugins.
- Tool calling: the manifest must declare `capabilities.tool_parser`; the launcher emits
  `--enable-auto-tool-choice --tool-call-parser <name>` (vLLM normalizes `_` to `-`).
- `serve` passes unknown args through to vLLM (for example `--port 8001` is `serve`'s own; see
  cli.md); `--print` echoes the resolved command and env without launching; `--local-only`
  requires an installed bundle and does not hit the Hub.
- **Updating:** `serve` warns (best-effort, 3 s bounded; `--no-update-check` to skip) when a newer
  revision exists. The update path is a plain `tt-model pull <id>`, which reinstalls a stale bundle
  in place. Do **not** tell users to `pull --force` to update: `--force` also skips the
  compat and wheel gates and is only for reinstalling regardless or overriding a warning.

### When something's wrong

Prefer a clear, actionable message over a silent workaround. If `pull` refuses (glibc,
interpreter, or arch), the fix is to **repackage on the right OS** rather than to force past the gate.
Consult the Troubleshooting table in [docs/E2E_RECIPE_V6.md](docs/E2E_RECIPE_V6.md) (v6) or
[docs/E2E_RECIPE_V5.1.md](docs/E2E_RECIPE_V5.1.md) (container). One v6 failure deserves a
specific note: a resolve error naming `certifi` (or `urllib3`, `requests`, `idna`) and
`download.pytorch.org` means the PyTorch CPU index captured that package. Fix the packaging (see
the recipe); never drop `--constraints` to make it "work" — without them the same install succeeds
silently with a stale CA bundle.
