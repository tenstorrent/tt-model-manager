# Self-contained (v5) model packages

> For the copy-paste, step-by-step walkthrough, see **[E2E_RECIPE.md](E2E_RECIPE.md)**
> (package, push, pull, serve). This document is the design and testing reference.

A **self-contained bundle** ships the platform *inside* the package: the author's built `ttnn`
wheel (custom C++ and low-level kernel (LLK) code compiled in), optionally the base vLLM and
plugin wheels, and their modified `tt-metal-community` tree, plus a generated `install.sh`,
`run.sh`, and a v5 manifest. Weights stay a **pointer** (a Hugging Face (HF) repo id), downloaded
at pull. A consumer needs only a Tenstorrent PCIe card and its firmware. `tt-model` alone does the
whole job, with no `tt-cli` and no pre-provisioned TT-Metalium™ or vLLM on the host.

## Layout and why this shape

One HF **model** repo carries everything needed to run. The rule is *"package what's on your
host"*:

```
wheels/            the author's built ttnn wheel (custom C++/LLK kernels compiled in),
                   base vLLM + plugin wheels, and the vendored dependency closure   (git-LFS)
metal/             the author's modified tt-metal-community tree (Python blocks + model code)
vllm_models/<name>/vllm_metadata.json     the plugin's EXTRA_MODELS_DIR contract
install.sh  run.sh  requirements.txt      generated launcher + installer
tt_kernel_manifest.json                   the v5 manifest
# weights: NOT embedded. A pointer (HF repo id) in the manifest, fetched on pull/serve
```

- **Self-contained.** The engine that runs is the author's actual build (their kernels ride
  along inside the `ttnn` wheel) rather than a stock pin, so custom C++ and LLK kernels work
  without extra steps.
- **Everything lives under the install directory.** After `pull`, everything needed to serve is
  under that directory: the pinned interpreter (provisioned by `uv` into the folder), the venv,
  the engine, the model code, and, on first serve, the weights and all caches. Serving depends on
  nothing outside the folder except the Tenstorrent device and system libc. Only `pull` touches
  the network.
- **Portable across machines.** The `ttnn` wheel is made portable with `auditwheel` (vendored
  libs and `$ORIGIN` RPATH). Deps are vendored so install is offline and reproducible. `pull`
  verifies the wheel's interpreter, arch, and **glibc** floor and fails clearly on a mismatch
  instead of at runtime. Build the engine wheel on **Ubuntu 22.04 (glibc 2.35)** and one bundle
  runs on both 22.04 and 24.04 (details under "Reproducibility and isolation" below).
- **Weights stay a pointer.** They are large and shared, so the manifest references the HF repo id.

## User flow

### Producer: package what's on your host
Build or bring up your model on `tt-metal-community` (your ttnn wheel now carries your kernels),
then:

```bash
tt-model package <your-org>/<model-name> \
  --from-metal . \
  --ttnn-wheel dist/ttnn-*.whl \
  --vllm-wheel dist/vllm-*.whl \
  --plugin-wheel dist/vllm_tt_plugin-*.whl \
  --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct \
  --mesh P150 \
  --max-num-seqs 32 --block-size 64 --max-model-len 4096
```

| Flag | Meaning |
|---|---|
| `<your-org>/<model-name>` | the HF target, a **positional** argument. Omit it and pass `--out <dir>` to stage locally |
| `--from-metal .` | your modified `tt-metal-community` tree |
| `--ttnn-wheel` | your built engine wheel (required) |
| `--vllm-wheel` | *(optional)* empty-target base vLLM |
| `--plugin-wheel` | *(optional)* the Tenstorrent vLLM plugin |
| `--arch-name` | HF architecture, written to `vllm_metadata` |
| `--weights` | **pointer** to the weights. The weights are not embedded |
| `--max-num-seqs`, `--block-size`, `--max-model-len` | serving args. The Tenstorrent backend needs a supported batch and block size |

`--wheels-dir <dir>` auto-classifies `ttnn-*` / `vllm-*` / `vllm_tt_plugin-*` instead of the
explicit flags. Large wheels go to Git Large File Storage (LFS) automatically on push. The result
is one HF **model** repo (the "running folder"): `wheels/`, `metal/`, `install.sh`, `run.sh`, a
per-model `vllm_models/<name>/vllm_metadata.json` (the EXTRA_MODELS_DIR contract), and
`tt_kernel_manifest.json`. If you omit `--max-num-seqs`/`--block-size`, the launcher defaults to
32/64 (the known-good tt_transformers values).

The embedded `metal/` tree is your source rather than your build. It **excludes** VCS (`.git`),
byte-caches (`__pycache__`, `*.pyc`), virtualenvs (`venv`, `.venv`), logs, and, at the tree root,
the regenerable multi-GB caches and build output (`.cpmcache`, `python_env`, `tt_cache`,
`build`/`build_*`, `built`, `built_kernels`). Symlinks are then normalized so the shipped tree is
self-contained (dangling links dropped, links escaping the tree materialized as real copies). The
kernels ship in your `ttnn` wheel, so none of the excluded build state is needed at serve.

### Consumer: pull and serve (only a card and firmware required)
```bash
tt-model pull  <org>/<model-name>     # installs the shipped wheels into the bundle's OWN venv,
                                        # (optionally --with-weights) downloads the weights
tt-model serve <org>/<model-name>     # runs the bundle's run.sh in that venv (OpenAI endpoint)
```
`serve` also install-then-serves a not-yet-pulled bundle. Everything runs from the bundle's venv.
Any TT-Metalium or vLLM on the host is not touched.

## Testing

### Offline (no hardware, no network)
The producer and consumer logic is fully unit-tested with mocked pip and HF:
```bash
pytest tests/test_v5.py                    # manifest v5 schema, self-contained compare() rules
pytest tests/test_packaging.py             # stage_package layout, wheel-tag parsing, CLI stage-only
pytest tests/test_self_contained_install.py # pull installs into a venv; serve runs run.sh
pytest                                     # full offline suite (no hardware, no network)
```

### Hardware smoke (a Tenstorrent card)
Validates the real round-trip. Stage locally, install, and serve:
```bash
# 1. stage a bundle from a built ttnn wheel + a metal-community tree
tt-model package --from-metal <community-clone> --ttnn-wheel <ttnn.whl> \
  --arch blackhole --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct --mesh P150 --out /tmp/bundle

# 2. install the shipped platform into the bundle's own venv
bash /tmp/bundle/install.sh /tmp/bundle/venv

# 3. sanity: the bundle venv opens the device (find_spec avoids the import-before-preload trap)
TTNN=$(/tmp/bundle/venv/bin/python -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("ttnn").origin))')
# a repaired wheel keeps _ttnncpp.so in ttnn.libs/ (RPATH target); a raw one in build/lib/. Prefer the former
PRELOAD=$(ls "$TTNN"/../*.libs/_ttnncpp*.so 2>/dev/null | head -1); [ -n "$PRELOAD" ] || PRELOAD=$(ls "$TTNN"/build/lib/_ttnncpp*.so | head -1)
LD_PRELOAD=$PRELOAD TT_METAL_HOME=$TTNN TT_METAL_VISIBLE_DEVICES=0 \
  /tmp/bundle/venv/bin/python -c "import ttnn; d=ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print(d.arch()); ttnn.close_mesh_device(d)"
```
**Expected:** `Arch.BLACKHOLE`, clean close. A full generation (the tt_transformers demo run from
the bundle venv) produced coherent text at about 75 tokens/s per user on a single Blackhole® p150
when the v5 path was validated.

### Serve-path requirements the tests encode

| Requirement | Why | Where it is enforced |
|---|---|---|
| The shipped `ttnn` wheel bundles `_ttnncpp.so`. It is pinned to interpreter, ABI, and arch (cp312/linux_x86_64). | `pull` refuses a wheel whose tags do not match the bundle's pinned interpreter or the host's arch and glibc floor. The host's own Python is not consulted; `install.sh` provisions the pinned one. | `host_incompatible_wheels` in `packaging.py` |
| Locate ttnn via `importlib.util.find_spec`, not `import ttnn`, when computing `LD_PRELOAD`. | The import is exactly what the preload fixes (glibc static thread-local storage (TLS)). | `run.sh` |
| Single-chip runs disable fabric and set `TT_METAL_VISIBLE_DEVICES=0`. | | `run.sh` |
| `vllm_metadata.json` lives in a per-model subfolder under `EXTRA_MODELS_DIR` (`vllm_models/<name>/`) rather than the bundle root. | The plugin scans children; a root-level file registers 0 architectures. | `stage_package` |
| `HF_MODEL` is exported for serving. | The tt_transformers adapter reads it from the env, not from vLLM's `--model`. `run.sh` sets both. | `run.sh` |
| Dependency pins satisfy **both** vLLM and `tt_transformers` under uv's *strict* resolver. | pip is lenient and hid conflicts. For example, `tt_transformers` pinning `pydantic==2.9.2` conflicts with vLLM's floor; relax such exact pins to a floor (`pydantic>=2.9.2`) in the metal tree's `requirements.txt`. `--vendor-deps` resolves the whole closure (platform wheels and requirements) together at package time, so a conflict surfaces on the producer, not at the consumer's install. | `--vendor-deps` |

## Validated end-to-end
The full loop (`package`, push to HF, `pull` to install the platform into the bundle venv,
`serve` the vLLM OpenAI endpoint, `curl` a chat completion) was run on a Blackhole p150 and
returned coherent output. The requirements in the table above (metadata subfolder, `HF_MODEL`
export, serving-arg defaults, `_ttnncpp.so` in the wheel, `find_spec` preload) each correspond to
a failure found in that run.

## Compatibility: the old `tt-kernel` name still works
The command was renamed from `tt-kernel` to `tt-model`. For anyone already on the old name:
- **`tt-kernel …` still runs** (a deprecated console-script alias for `tt-model`). It prints a
  one-line note to stderr pointing at `tt-model`.
- **`TT_KERNEL_*` env vars are still honored** as a fallback for their `TT_MODEL_*` replacements.
- **An existing `~/.cache|.config/tt-kernel` dir is reused** when the new `tt-model` one does not
  exist yet, so a pre-rename install keeps finding its bundles and index.
- **Bundles published with the old tool still install** *if they are v5/v6*. The on-disk manifest
  filename is unchanged (`tt_kernel_manifest.json`); pre-v5 schemas are refused (re-publish with a
  current `tt-model`).

These shims are deprecated; switch to `tt-model` when convenient.

## Reproducibility and isolation (why push-here/pull-there works)
Each of the following guarantees closes a way in which an artifact could depend on the author's
host without anyone noticing:

- **Portable engine wheel.** `tt-model package` runs `auditwheel repair` on the ttnn wheel
  (`--repair`, default). It vendors the external libs (libtracy, libmpi, libhwloc, libnuma, and
  their deps) into `ttnn.libs/` and rewrites RPATH to `$ORIGIN`. Without repair, the shipped
  `.so` files' RPATH leads with the build tree, so on the build host everything loads from there
  and validation exercises the build tree rather than the artifact. Repair requires `auditwheel`
  and `patchelf` on the author's host, and the build tree present so the libs can be found.
  `--no-repair` opts out with a loud warning.
- **Pinned interpreter and offline deps.** `install.sh` uses **uv** to provision the *exact*
  Python (the host Python does not have to match) and installs **offline from the vendored
  dependency wheels** (`--vendor-deps`, default), so there is no PyPI or resolver drift and no
  network at install. The pinned version and vendored flag are recorded in the manifest.
- **Hermetic install.** The interpreter is provisioned *into the bundle*
  (`UV_PYTHON_INSTALL_DIR=$HERE/.python`), the venv is built `--relocatable` with
  `--link-mode=copy` (package contents copied in rather than hardlinked to uv's global cache), and
  `run.sh` redirects every cache and home under `$HERE`: `HF_HOME`,
  `TT_CACHE_PATH`/`TT_CACHE_HOME` (the ttnn tensor cache defaults to a hard-coded `/mnt/...`
  path upstream, which this overrides), `XDG_CACHE_HOME`, triton, and inductor. After install,
  serving depends on nothing outside the folder except the Tenstorrent device and system libc.
  Each redirect is overridable (`${VAR:-$HERE/...}`).
- **glibc floor is checked.** The repaired wheel is tagged for the glibc of the host it was
  repaired on (`manylinux_2_39` on Ubuntu 24.04). `pull` compares that floor to the host glibc and
  fails with a clear message on a too-old host, instead of a cryptic `GLIBC_2.39 not found` at
  dlopen. **Build the engine wheel on Ubuntu 22.04 (glibc 2.35)**; it then repairs to
  `manylinux_2_35` and one bundle runs on both 22.04 and 24.04. `package --manylinux <policy>`
  asserts a floor at build time.
- **serve checks for updates.** When an installed bundle is served, `serve` compares the
  installed revision to the Hub's current tip (one short request with a 3 s timeout, so it cannot
  hang the serve) and prints a non-blocking advisory if a newer one exists, pointing at a plain
  `tt-model pull <id>`. Skipped for a pinned `@revision` install, under `--local-only`, and with
  `--no-update-check`.
- **Updating does not need `--force`.** The recorded revision is the exact commit `pull` fetched
  (resolved *before* the download, so it always matches what is on disk). A plain
  `tt-model pull <id>` **reinstalls a stale bundle in place** (and reuses an up-to-date one).
  `--force` is only for reinstalling regardless, or pushing past a compat or wheel warning. It is
  *not* the update path, and it also skips those safety gates.
- **No `VLLM_PLUGINS` in run.sh.** That variable is an allow-list; setting it silently suppresses
  the model's tool and reasoning-parser plugins. The Tenstorrent platform and registry load via
  entry points.
- **`_ttnncpp` preload is located wherever it lives** (`ttnn.libs/` for a repaired wheel, else
  `build/lib/`). `run.sh` globs both.
- **Re-pull preserves your edits.** An up-to-date install is reused (local edits to `run.sh` and
  similar are kept); a newer revision reinstalls in place; `--force` reinstalls regardless.
- **serve pass-through and `--print`.** `tt-model serve <id> <vllm args>` forwards extra args
  (see the pass-through rule in [cli.md](cli.md#run-a-model)); `--print` echoes the fully
  *resolved* command and env rather than a bare `bash run.sh`.
