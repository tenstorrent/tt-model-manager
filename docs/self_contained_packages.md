# Self-contained (v5) model packages

> For the copy-paste, step-by-step walkthrough, see **[E2E_RECIPE.md](E2E_RECIPE.md)**
> (model → package → push → pull → serve). This document is the design + testing reference.

A **self-contained bundle** ships the platform *inside* the package: the author's built `ttnn`
wheel (custom C++/LLK kernels compiled in), optionally the base vLLM + plugin wheels, and their
modified `tt-metal-community` tree — plus a generated `install.sh`/`run.sh` and a v5 manifest.
Weights stay a **pointer** (an HF repo id), downloaded at pull. A consumer needs only a TT card +
firmware. `tt-model` alone does the whole job — no tt-cli, no pre-provisioned tt-metal/vLLM.

## User flow

### Producer — "package what's on your box"
Build/bring up your model on `tt-metal-community` (your ttnn wheel now carries your kernels), then:

```bash
tt-model package <your-org>/<model-name> \        # HF target is a POSITIONAL arg (omit + --out to stage locally)
  --from-metal .                                   # your modified tt-metal-community tree
  --ttnn-wheel dist/ttnn-*.whl                      # your built engine wheel (required)
  --vllm-wheel dist/vllm-*.whl                      # optional: empty-target base vLLM
  --plugin-wheel dist/vllm_tt_plugin-*.whl          # optional: the TT vLLM plugin
  --arch-name LlamaForCausalLM                      # HF architecture -> vllm_metadata
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct           # POINTER — weights are not embedded
  --mesh P150 \
  --max-num-seqs 32 --block-size 64 --max-model-len 4096   # serving args (TT backend needs a
                                                            # supported batch + block size)
```
`--wheels-dir <dir>` auto-classifies `ttnn-*` / `vllm-*` / `vllm_tt_plugin-*` instead of the
explicit flags. Large wheels go to git-LFS automatically on push. The result is one HF **model**
repo (the "running folder"): `wheels/`, `metal/`, `install.sh`, `run.sh`, a per-model
`vllm_models/<name>/vllm_metadata.json` (the EXTRA_MODELS_DIR contract), and
`tt_kernel_manifest.json`. If you omit `--max-num-seqs`/`--block-size`, the launcher defaults to
32/64 (the known-good tt_transformers values).

### Consumer — pull + serve (only a card + firmware required)
```bash
tt-model pull  <org>/<model-name>     # installs the shipped wheels into the bundle's OWN venv,
                                        # (optionally --with-weights) downloads the weights
tt-model serve <org>/<model-name>     # runs the bundle's run.sh in that venv (OpenAI endpoint)
```
`serve` also install-then-serves a not-yet-pulled bundle. Everything runs from the bundle's venv;
the host's tt-metal/vLLM (if any) is never touched.

## Testing

### Offline (no hardware, no network)
The producer/consumer logic is fully unit-tested with mocked pip + HF:
```bash
pytest tests/test_v5.py                    # manifest v5 schema, self-contained compare() rules
pytest tests/test_packaging.py             # stage_package layout, wheel-tag parsing, CLI stage-only
pytest tests/test_self_contained_install.py # pull installs into a venv; serve runs run.sh
pytest                                     # full offline suite (no hardware, no network)
```

### Hardware smoke (a TT card)
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
# a repaired wheel keeps _ttnncpp.so in ttnn.libs/ (RPATH target); a raw one in build/lib/ — prefer the former
PRELOAD=$(ls "$TTNN"/../*.libs/_ttnncpp*.so 2>/dev/null | head -1); [ -n "$PRELOAD" ] || PRELOAD=$(ls "$TTNN"/build/lib/_ttnncpp*.so | head -1)
LD_PRELOAD=$PRELOAD TT_METAL_HOME=$TTNN TT_METAL_VISIBLE_DEVICES=0 \
  /tmp/bundle/venv/bin/python -c "import ttnn; d=ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print(d.arch()); ttnn.close_mesh_device(d)"
```
**Expected:** `Arch.BLACKHOLE`, clean close. A full generation (the tt_transformers demo run from
the bundle venv) produces coherent text at ~75 tok/s/user on a single p150.

### Notes / gotchas the tests encode
- The shipped `ttnn` wheel **must** bundle `_ttnncpp.so`; it's py/abi/arch-pinned (cp312/linux_x86_64),
  and `pull` refuses a wheel that doesn't match the host interpreter (`host_incompatible_wheels`).
- Locate ttnn via `importlib.util.find_spec`, never `import ttnn`, when computing `LD_PRELOAD` — the
  import is exactly what the preload fixes (glibc static-TLS). `run.sh` does this.
- Single-chip: fabric disabled + `TT_METAL_VISIBLE_DEVICES=0`.
- **`vllm_metadata.json` must be in a per-model subfolder** under `EXTRA_MODELS_DIR` (`vllm_models/<name>/`),
  not the bundle root — the plugin scans children, so a root-level file registers 0 architectures.
- **`HF_MODEL` must be exported** for serving: the tt_transformers adapter reads it from the env, not
  from vLLM's `--model` (both are set by `run.sh`).
- Dependency pins must satisfy **both** vLLM and `tt_transformers` under uv's *strict* resolver
  (pip is lenient and hid this). E.g. `tt_transformers` pinning `pydantic==2.9.2` conflicts with
  vLLM's floor — relax such exact pins to a floor (`pydantic>=2.9.2`) in the metal tree's
  `requirements.txt`. `--vendor-deps` resolves the whole closure (platform wheels + requirements)
  together at package time, so a conflict surfaces on the producer, not at the consumer's install.

## Validated end-to-end
The full loop — `package` → push to HF → `pull` (install-the-platform into the bundle venv) →
`serve` (vLLM OpenAI endpoint) → `curl` a chat completion — was run on a Blackhole p150 and
returned coherent output. The fixes above (metadata subfolder, `HF_MODEL` export, serving-arg
defaults, `_ttnncpp.so` in the wheel, `find_spec` preload) all come from that run.

## Compatibility: the old `tt-kernel` name still works
The command was renamed `tt-kernel` → `tt-model`. For anyone already on the old name:
- **`tt-kernel …` still runs** (a deprecated console-script alias for `tt-model`); it prints a
  one-line note to stderr pointing at `tt-model`.
- **`TT_KERNEL_*` env vars are still honored** as a fallback for their `TT_MODEL_*` replacements.
- **An existing `~/.cache|.config/tt-kernel` dir is reused** when the new `tt-model` one doesn't
  exist yet, so a pre-rename install keeps finding its bundles/instances/index.
- **Bundles published with the old tool still install** — the on-disk manifest is unchanged
  (`tt_kernel_manifest.json`), and every prior schema version is still read.

These shims are deprecated; switch to `tt-model` when convenient.

## Reproducibility & isolation (why push-here/pull-there works)
Cross-machine review found the artifact was never truly self-contained — it silently leaned on the
author's box. These are now closed by construction:

- **Portable engine wheel.** `tt-model package` runs `auditwheel repair` on the ttnn wheel
  (`--repair`, default): it vendors the external libs (libtracy/libmpi/libhwloc/libnuma + their
  deps) into `ttnn.libs/` and rewrites RPATH to `$ORIGIN`. Before this, the shipped `.so`s' RPATH
  led with the build tree, so on the build box everything loaded from there and validation tested
  the build tree, not the artifact. (Requires `auditwheel`+`patchelf` on the author's box, and the
  build tree present so the libs can be found; `--no-repair` opts out with a loud warning.)
- **Pinned interpreter + offline deps.** `install.sh` uses **uv** to provision the *exact* Python
  (the host Python no longer has to match) and installs **offline from the vendored dependency
  wheels** (`--vendor-deps`, default) — no PyPI/resolver drift, no network at install. The pinned
  version and vendored flag are recorded in the manifest.
- **Hermetic — the folder is the wall.** The interpreter is provisioned *into the bundle*
  (`UV_PYTHON_INSTALL_DIR=$HERE/.python`), the venv is built `--relocatable` with `--link-mode=copy`
  (package contents copied in, not hardlinked to uv's global cache), and `run.sh` redirects every
  cache/home under `$HERE` — `HF_HOME`, `TT_CACHE_PATH`/`TT_CACHE_HOME` (the ttnn tensor cache
  *defaults to a hard-coded `/mnt/...` upstream* — a classic other-machine leak we override),
  `XDG_CACHE_HOME`, triton/inductor. After install, serving depends on nothing outside the folder
  except the TT device + system libc. Each redirect is overridable (`${VAR:-$HERE/...}`).
- **glibc floor is checked.** The repaired wheel is tagged for the glibc of the box it was repaired
  on (`manylinux_2_39` on Ubuntu 24.04). `pull` compares that floor to the host glibc and fails with
  a clear message on a too-old host, instead of a cryptic `GLIBC_2.39 not found` at dlopen. **Build
  the engine wheel on Ubuntu 22.04 (glibc 2.35)** — then it repairs to `manylinux_2_35` and one
  bundle runs on both 22.04 and 24.04. `package --manylinux <policy>` asserts a floor at build time.
- **serve checks for updates.** When an installed bundle is served, `serve` compares the installed
  revision to the Hub's current tip and prints a non-blocking advisory if a newer one exists
  (skipped for a pinned `@revision` install and under `--local-only`).
- **No `VLLM_PLUGINS` in run.sh.** That variable is an allow-list; setting it silently suppressed
  the model's tool/reasoning-parser plugins. The TT platform + registry load via entry points.
- **`_ttnncpp` preload is located wherever it lives** (`ttnn.libs/` for a repaired wheel, else
  `build/lib/`) — run.sh globs both.
- **Re-pull preserves your edits** — an existing install is reused unless `--force`.
- **serve pass-through + `--print`.** `tt-model serve <id> -- <vllm args>` forwards extra args;
  `--print` echoes the fully *resolved* command+env (not a bare `bash run.sh`).
