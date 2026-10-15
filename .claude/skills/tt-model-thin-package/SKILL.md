---
name: tt-model-thin-package
description: Package a model as a v6 THIN bundle the canonical way and prove it works — stock ttnn from PyPI, tt-transformers as tooling only, the model's own code shipped inside the bundle (--extra-code), custom kernels only as generic_op wheels — then install it, import it, serve it on hardware, verify the API (tool calling included), and push only when asked. Use for "package this model as v6", "make a thin bundle", "ship this model without docker". Do NOT use for v5.1 containers (use tt-model-package-test), and do NOT use it to contribute a model into tt-transformers.
---

# Package a model as a v6 thin bundle, the canonical way

Input: a model that already serves on your host (its model code, any custom kernels, and the
weights id). Output: a staged — and, when asked, published — v6 bundle, plus a report that says
exactly what was verified and what was not. **Never claim the bundle works without Step 5.**

## The canonical shape

```
<bundle>/                      schema 6, `deps` block
  model.py                     entry point: imports your ops wheel, re-exports the vLLM adapter class
  extra_code/<model_pkg>/      YOUR model's code (--extra-code); run.sh puts extra_code/ first on PYTHONPATH
  requirements.txt             ttnn==<version on PyPI> (+ torchvision). NO index options in this file.
  constraints.txt              fully resolved pins for every install step (--constraints)
  wheels/
    <model>_ops-*.whl          custom kernels via ttnn.generic_op (--ops-wheel) — only if you need any
    tt_transformers-*.whl      TOOLING ONLY, no models/ (until tt-transformers publishes to PyPI)
    vllm_tt_plugin-*.whl       (--plugin-wheel)
    vllm-*+empty-*.whl         prebuilt empty-target vLLM (--vllm-wheel)
```

Each layer has one right answer. The wrong answers all exist in published bundles:

| layer | canonical | do NOT |
|---|---|---|
| engine | **stock `ttnn` from PyPI** | ship a forked or custom-built `ttnn` (a `+local` version tag is the tell) |
| custom kernels | a **`generic_op` wheel** via `--ops-wheel` | compile kernels into `ttnn` |
| tooling | **tt-transformers, tooling only** | contribute your model *into* tt-transformers; ship a tt-transformers wheel that still carries `models/` |
| model code | **inside the bundle**, via `--extra-code` | pin a whole-`models/`-tree wheel (e.g. `tt-metal-models`) to run one model — it ships every other model's code with yours |
| dependency safety | **`--constraints` with fully resolved pins** | rely on index order; put `--extra-index-url` in `requirements.txt` |
| serving capabilities | **declare `--tool-parser` / `--reasoning-parser`** | leave them unset (tools silently unavailable; reasoning mixed into `content`) |

Why tt-transformers is tooling only: models are being moved out of it, so a model contributed
into it has no future home there. Depend on its tooling API — in practice
`modules.lightweightmodule`, `sampling`, `tensor_utils`, `device_utils`, `modules.tt_ccl` — and
ship your model yourself.

## Step 1 — Preflight (no card, minutes)

Answer each before building anything. A "no" is a stop-and-fix, not a footnote.

1. **Is every custom kernel a `generic_op` wheel on stock ttnn?** If your bring-up needed a
   modified ttnn, move those kernels into a `generic_op` wheel first.
2. **Does your model package import only tooling from tt-transformers?** List its imports from
   `tt_transformers.*` outside its own package; anything under `tt_transformers.models.*` means
   it still depends on another model.
3. **Does any tooling module import your model back?** It must not, or a tooling-only wheel breaks.
4. **Do you have a fully resolved `constraints.txt`?** Without one, Step 4's install can silently
   resolve stale networking packages (see Known issues).
5. **What does the model need at serve time?** Max context, sequences, block size, the
   `tt` additional config (fabric, trace region), env, tool and reasoning parsers. Copy these
   from the configuration you have actually validated; do not guess.

## Step 2 — Lay out the inputs

**2a. Tooling-only tt-transformers wheel** (until tt-transformers publishes). From the wheel you
develop against: delete everything in `tt_transformers/models/` except `__init__.py`, remove any
`Requires-Dist` on your model's own packages (e.g. your ops wheel), drop the stale `RECORD`, and
repack with `python -m wheel pack`.

**2b. Move your model into `extra_code/<model_pkg>/`.** Rewrite its absolute self-imports
(`tt_transformers.models.<x>` → `<model_pkg>`), then check three things a rewrite can miss:
- **dynamic imports** — `importlib`, `__import__`, or module paths in strings; the rewrite
  does not touch them, so there must be none (or fix them by hand)
- **data files** — YAML/JSON next to the code must move with it
- **`__file__`-relative paths** — e.g. `Path(__file__).parents[3]` counts directories, so it
  changes meaning after a move; confirm each one is either relative to its own file or never reached

**2c. Prove every import resolves, statically.** Walk every `import` / `from … import` in
`extra_code/` and `model.py` (relative ones included) and confirm each target exists in
*`extra_code/` ∪ the tooling wheel*. Zero unresolved, or stop.

**2d. `model.py`** — import your ops wheel explicitly, then re-export the adapter:

```python
import <model>_ops  # noqa: F401  — declares the custom-op dependency (deps.custom_ops reads model.py)
from <model_pkg>.<adapter_module> import <AdapterClass>
__all__ = ["<AdapterClass>"]
```

The explicit import is not decoration: `deps.custom_ops` is recorded from `model.py`'s own
imports, so without it the bundle declares no custom surface.

**2e. `requirements.txt`** — `ttnn==<version>` plus anything the venv needs from an index, and
**no `--extra-index-url` or other index option** (pip and uv honour them inside the file).
**`constraints.txt`** — pin every resolved package, including `certifi`, `urllib3`, `requests`.

**2f. Install guards (`--verify`)** — fail the install, not the first request. At minimum assert:
the ttnn version; that the shipped tt-transformers has no `models/<model>`; that your model's
code is present under `extra_code/`. `install.sh` runs them with the venv's python and does not
export `$HERE`, so locate the bundle from `sys.prefix` (the venv lives at `<bundle>/venv`). Test
each guard against a layout it should **reject** before relying on it.

## Step 3 — Stage locally (no repo id) and inspect

```bash
tt-model package-thin --out <dir> --name <bundle-name> \
  --model-py model.py --extra-code extra_code \
  --requirements requirements.txt --constraints constraints.txt --python 3.12.14 \
  --plugin-wheel wheels/vllm_tt_plugin-*.whl \
  --ops-wheel    wheels/<model>_ops-*.whl \
  --ops-wheel    wheels/tt_transformers-*.whl \
  --vllm-wheel   wheels/vllm-*+empty-*.whl --vllm-version <v> --vllm-common-requirements vllm-common.txt \
  --arch blackhole --arch-name <HFArch> --main-class model:<AdapterClass> \
  --weights <org>/<weights> --weights-revision <sha> \
  --mesh <topology> --device-count <n> --mesh-fabric <FABRIC_1D|…> \
  --max-model-len <n> --max-num-seqs <n> --block-size <n> --trace-region-bytes <n> \
  --tt-config-json '<json>' --tool-parser <name> --reasoning-parser <name> \
  --env KEY=VALUE … --extra-arg <token> … \
  --verify '<statement>' --provenance provenance.json
```

**Omit the positional `<org>/<name>`.** With it, `package-thin` stages *and* pushes.
`--provenance` takes a **path** to a JSON file. Save the full command as a script so the bundle
can be rebuilt exactly.

Then inspect the staged output — this is where most mistakes are caught cheaply:
- **the vLLM argv** — expand the `CMD=(…)` line in `run.sh` through bash and check every JSON
  argument (`--additional-config`, generation config, chat-template kwargs) arrives as **one** token
- **the env** — every variable your validated configuration needs, and nothing contradicting it
- **`deps.custom_ops`** in the manifest — lists exactly your ops wheel(s), never the tooling wheel
- **`run.sh` PYTHONPATH** — `extra_code/` first

## Step 4 — Prove it installs and imports (no card)

**4a.** Copy the staged folder and run its `install.sh` there (never in the staged folder you will
push — it creates `venv/` and caches). Pass: exit 0, `All installed packages are compatible`, and
every `--verify` guard ran. Confirm the shipped tt-transformers' `models/` holds only `__init__.py`.

**4b. Import smoke test.** With the same `LD_PRELOAD` `run.sh` computes (ttnn's `_ttnncpp.so`,
found via `importlib.util.find_spec`, never `import ttnn`) and
`PYTHONPATH=<copy>/extra_code:<copy>`, run `import model` and check: the adapter class loads from
`extra_code/`, and **zero** modules were imported from the model's old location. This catches an
import error in seconds instead of after a multi-minute boot.

## Step 5 — Serve on hardware and prove it works

`tt-model serve` does not take a local v6 bundle (it expects a pulled `<org>/<name>`), so serve the
installed copy with its own launcher:

```bash
bash <copy>/run.sh --port <port>
```

The first start loads weights, builds the tensor cache, and JIT-compiles kernels — minutes (often
10+ on a cold cache). Wait for `Application startup complete`. If the boot dies with
`Timed out while waiting for active ethernet core`, the devices are dirty, not the bundle: confirm
nothing else holds them, `tt-smi -r`, and serve again. Reset only on that evidence.

Then run every check that applies — identical to the container path:
1. **Identity** — `GET /v1/models` returns the manifest's weights id.
2. **Generation** — a `temperature: 0` completion returns coherent text. Give reasoning models
   room (≥1500 `max_tokens`): a small budget returns `content: null` with the text in `reasoning`.
3. **Tool calling** — REQUIRED when the manifest declares a tool parser: a real `tools` request
   must return `finish_reason: tool_calls` with arguments that are valid JSON.
4. **Reasoning** — with a reasoning parser, `reasoning` is populated separately from `content`.
5. **Reference equivalence** — if a known-good build of this model exists (a v5.1 container, an
   earlier bundle), compare a `temperature: 0` completion: same text is the strongest evidence the
   packaging changed nothing. Under speculative decoding, also compare tokens-per-chunk.

## Step 6 — Stop cleanly

Stop the server with SIGTERM (Ctrl-C in its terminal, or `kill -INT` to its **process group** —
vLLM forks `EngineCore` children that outlive their parent). Confirm no `vllm` / `EngineCore`
process remains before the next serve.

## Step 7 — Push (only when the user asks), then pull and re-prove

Before pushing, **look at the target repo**. If it exists, check what bundle it holds
(`tt_kernel_manifest.json` `name`) and its visibility; pushing replaces it. A new repo is created
private. Pass `--private` / `--public` only when asked — on an existing repo either flag
**changes** its visibility.

Push by re-running the Step 3 command **with** the positional `<org>/<name>` (`tt-model push` is
for v5.1 containers). Then prove the published copy, not the local one:

```bash
tt-model pull <org>/<name> --with-weights
tt-model serve <org>/<name> --local-only --no-update-check --port <port>
```

Re-run Step 5's checks and confirm the result matches Step 5 (same greedy text). Stop with
`tt-model stop <org>/<name>`.

## Known tt-model issues this pattern works around

Remove each note when its fix lands.

- **The requirements step ranks the PyTorch CPU index above PyPI.** uv then takes any package that
  index carries — `certifi`, `urllib3`, `requests`, `idna` — *only* from it, at a single years-old
  version. With `--constraints` the install fails loudly; **without, it silently installs a stale
  CA bundle and old `requests`/`urllib3`**. That is why `--constraints` is required here, and why
  the tooling wheel goes through `--ops-wheel` (installed by path after vLLM) rather than
  `--models-wheel`. Once tt-transformers publishes, do not pin it in `requirements.txt` until this
  is fixed.
- **`deps.custom_ops` reads only `model.py`.** Hence the explicit ops import in Step 2d.
- **Serving a local v6 path errors misleadingly** ("not a tt-model bundle"). Use `run.sh` (Step 5).

## Report

End with two lists, and be exact: **verified** (each Step 4 and Step 5 check that ran, on which
hardware) and **not verified** (everything else — other topologies, capabilities not exercised,
the published copy if Step 7 did not run). "It installed" is not "it works"; only Step 5 earns
that sentence.
