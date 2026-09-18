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
     builds the venv from pip pins (`ttnn` / `tt-metal-models`) plus bundled wheels, with no
     embedded `ttnn` wheel and no `metal/` tree. `deps.kind` picks the serving front end
     `render_run_sh` puts in `run.sh`: `"vllm"` (the default, and v6's only behavior before this
     field existed) adds an empty-target vLLM build step (the `vllm-tt-plugin` plus any
     `generic_op` wheel) and serves `vllm.entrypoints.openai.api_server`; `"tt-dit-server"` (no
     vLLM step regardless of `--vllm`/`--no-vllm`) serves `deps.app` directly with uvicorn
     instead, for a model with no tokens, KV cache, or continuous batching, mirroring the
     same-named v5.1 CONTAINER kind (`launchers.TtDitServerLauncher`).

   In every case the engine that serves is the one the bundle builds rather than a shared host install.
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
5. If the change touches the serve or device path, validate on hardware (see
   `docs/thin_packages.md`, Testing it in the lab, or the `tt-model-package-test` skill for a
   container package): package, pull, serve, `curl`.
6. Commit messages: imperative subject, a short body explaining *why*, and end with a
   `Co-Authored-By` trailer naming the assistant that produced the change, for example
   `Co-Authored-By: <assistant name> <noreply@anthropic.com>`.
7. Open a **draft PR**. The body states what and why, the test evidence (for example "full
   suite: N passed"), and a link to the tracking issue if one applies.

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
calling included), and pushes. Prefer them over improvising the flow.

There is one **venv-based authoring path**, also self-contained (a consumer needs only a card
and firmware):

- **v6 "thin"**: `tt-model package-thin …` builds the venv from pip pins (`ttnn` /
  `tt-metal-models`) plus bundled wheels (`vllm-tt-plugin` plus any `generic_op`). Gated on
  `tt-metal-models` publishing (`tt-metal#54478`) and `tt_transformers` being broken out; the
  generated `requirements.txt` ships with a TODO pin until then. Design:
  [docs/thin_packages.md](docs/thin_packages.md); copy-paste recipe:
  [docs/E2E_RECIPE.md](docs/E2E_RECIPE.md).

### The canonical sequence

1. **Bring up** the model on `tt-metal-community`.
2. **package-thin** (producer). Pushes to HF unless you pass `--out <dir>` to stage locally.
   Omit `--requirements` and it writes the template with the `tt-metal-models` TODO pin:
   ```
   tt-model package-thin <org>/<name> --model-py model.py --requirements requirements.txt \
     [--plugin-wheel …] [--ops-wheel …] --arch <isa> --arch-name <HFArch> \
     --main-class <module:Class> --weights <hf-id> --mesh <mesh>
   ```
3. **pull** (consumer): `tt-model pull <org>/<name>` (`--with-weights` to pre-download weights).
4. **serve** (consumer): `tt-model serve <org>/<name> [--port N] [--print] [--local-only]`.
5. **verify**: `curl .../v1/chat/completions` returns coherent text.

There is no `install`, `run`, or `start` command. `package-thin` pushes; `pull` installs;
`serve` installs then serves. `tt-model push` exists only for a staged v5.1 container
directory. See the pass-through rule in [docs/cli.md](docs/cli.md#run-a-model) for which `serve`
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
- The engine is what the bundle builds: v6 builds it from the pinned deps.

### Verification checkpoints: do not claim success without them

- **After package-thin (v6):** the bundle has `model.py`, `requirements.txt`, the bundled
  `vllm-tt-plugin` (plus any `generic_op`) wheel, `vllm_models/<name>/vllm_metadata.json`,
  `install.sh`, `run.sh`, `tt_kernel_manifest.json`.
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
Consult the Troubleshooting table in [docs/E2E_RECIPE.md](docs/E2E_RECIPE.md). If the
`tt-metal-models` pin cannot resolve yet, that is the expected gate. The path is not runnable until
`tt-metal#54478` publishes; see [docs/thin_packages.md](docs/thin_packages.md).
