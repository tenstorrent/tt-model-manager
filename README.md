# tt-model

> ⚠️ **Experimental — no support, no guarantees.** tt-model is an early, experimental
> project. Nothing here is officially supported, and we make no claim of correctness,
> stability, or fitness for any purpose. APIs, the bundle format, and behavior may change
> or break at any time without notice. Use it at your own risk.

`tt-model` distributes models over the Hugging Face Hub and serves them on Tenstorrent
hardware. The **default serving path is the Tenstorrent vLLM plugin** — an OpenAI-compatible
server. One command pulls a model and brings the server up:

```bash
tt-model serve <namespace>/<model>     # pull the bundle, register it with vLLM, launch the server
```

### vLLM bundles (the default)

A **vLLM bundle** is a small, self-contained folder: a plugin-owned `vllm_metadata.json`
(the HF architecture, the generator-adapter class, a per-machine launch command, and a
reference to the HF weights) plus the adapter code and its dependencies. It ships **no kernel
cache and no weights** — vLLM JIT-compiles kernels at first-run warmup into tt-metal's own
local cache, and the model fetches weights from the referenced HF repo. On `pull`, the folder
is placed into a local bundles directory that the vLLM plugin discovers via `EXTRA_MODELS_DIR`
and auto-registers, so no per-model edit to the plugin is needed. **Weights are never stored in
a bundle** — only referenced by their HF repo id. To author one, see
**[docs/authoring_runners.md](docs/authoring_runners.md)**.

### Kernel-cache bundles (legacy dispatch path)

`tt-model` also publishes and pulls **precompiled tt-metal kernel caches** for the older
"dispatch" serving runtime (`tt_api.serve`), so a model's first run is a cache **hit** instead
of a slow JIT recompile. tt-metal JIT-compiles every kernel on first run and caches the RISC-V
binaries; they are deterministic for a fixed `(tt-metal build, arch, device config,
compile-time args)` tuple. `tt-model` packages that cache, publishes it as an HF model repo
addressed `namespace/name`, and validates compatibility before installing it locally. See
[Kernel-cache bundles (legacy)](#kernel-cache-bundles-kernels--runner--weights-legacy-dispatch-path)
below.

`tt-model serve` (and `tt-model run`) is the **front door**: it resolves whether a bundle
exists and routes accordingly — the vLLM plugin by default, the dispatch runtime for a
kernel-cache bundle, or the dynamic runtime on a bare Hugging Face repo. Custom implementations
win; nothing overrides them.

## Install

From a fresh clone, one command sets up the whole serving stack — the Tenstorrent **vLLM
fork + plugin** plus `tt-model` — on top of a working tt-metal env, and verifies it:

```bash
scripts/install.sh           # bootstraps tt-model, then runs `tt-model install`
```

Once `tt-model` is on PATH, use the CLI directly:

```bash
tt-model install                                  # same thing, without the shim
tt-model install --venv <tt-metal>/python_env     # target a specific tt-metal env
tt-model install --verbose                        # stream pip instead of collapsing it
```

`install` expects tt-metal (`ttnn`) to already be importable in the target environment —
building it is out of scope — and **stops before installing anything** if it is not,
rather than spending ~450MB on an environment that could never serve a model. If `ttnn`
is missing it tells you the two ways to get it, one of which is just
`pip install "ttnn>=0.72"` from PyPI.

Exit codes: `0` installed and adequate · `1` preflight failed, nothing installed ·
`2` usage error · `3` installed, but the toolchain is still not adequate.

## Usage

```bash
tt-model install                                 # set up the serving stack, then verify
tt-model login                                   # reuses huggingface_hub's token store
tt-model doctor                                  # check tt-metal/vLLM + hardware

# vLLM (default) — serve a model through the Tenstorrent vLLM plugin
tt-model serve you/mymodel                        # pull if needed, register, launch the OpenAI server
tt-model serve you/mymodel --print                # print the launch command instead of running it
tt-model push  you/mymodel --backend vllm \       # publish a vLLM bundle from a v4 manifest
  --manifest ./model.json --bundle-dir ./adapter   # (--bundle-dir optional for built-ins)

# Shared / discovery
tt-model pull  you/mymodel                        # download + install a bundle locally
tt-model info  you/mymodel                        # manifest + compatibility verdict
tt-model search gemma                             # discover published bundles
tt-model search gemma --catalog                   # only bundles listed in the community catalog
tt-model search --target p150x4 --arch blackhole  # "what runs on my box" (v4 tags)
tt-model list                                     # locally installed bundles
tt-model rm    you/mymodel                        # remove an installed bundle

# tt-metal instances (which build serves a v4 model)
tt-model instances list --for you/mymodel         # installed builds + which satisfy the model
tt-model instances add --name metal-0.73 \        # register a build auto-scan can't find
  --python /opt/tt/0.73/venv/bin/python --tt-metal-home /opt/tt/0.73

# Kernel-cache (legacy dispatch path) — see below
tt-model run   you/smallmodel-blackholex1         # dispatch runtime + precompiled cache
tt-model clean --all                              # wipe cache subtrees for a clean producer state
```

## Serving with vLLM (the default)

`tt-model serve <id>` is the one-command path. It pulls the bundle folder if it isn't already
installed, lays it into the local bundles directory, points the vLLM plugin at that directory
via `EXTRA_MODELS_DIR`, and launches the OpenAI-compatible server with the bundle's
per-machine launch command:

```bash
tt-model serve you/mymodel                   # pull-if-needed -> register -> launch; prints the endpoint
tt-model serve you/mymodel --print           # emit the exact launch command + env instead of running
tt-model serve you/mymodel --local-only      # require an installed bundle; never hit the Hub
tt-model serve you/mymodel --bundles-dir DIR # override the EXTRA_MODELS_DIR location
```

Repeat invocations skip the pull and go straight to launch. `tt-model run <id>` routes a vLLM
bundle to this same path.

### Publishing a vLLM bundle

Author a bundle folder — a `vllm_metadata.json` plus the adapter class (or a reference to an
existing tt-metal generator) — then push it. It is **kernels-less**: no precompiled cache and
no weights are shipped. See **[docs/authoring_runners.md](docs/authoring_runners.md)** for the
metadata schema and the adapter contract.

```bash
# v4 (recommended): author one manifest; tt-model renders vllm_metadata.json on pull
tt-model push you/mymodel --private --backend vllm \
  --manifest ./model.json \                # unified manifest (entrypoint/platform/runtime/…)
  --bundle-dir ./adapter                    # optional: custom adapter class + extension wheels

# legacy: ship a hand-written vllm_metadata.json verbatim
tt-model push you/mymodel --private --backend vllm \
  --bundle-dir ./bundle --weights some-org/mymodel

tt-model pull you/mymodel                  # lay the folder into the local bundles dir
tt-model pull you/mymodel --with-weights   # ...and also pre-download the weights (default: skip)
```

The plugin auto-registers every bundle it finds under `EXTRA_MODELS_DIR`, so no per-model edit
to the plugin is required. `vllm_metadata.json` is owned by the plugin; `tt-model` ships it
verbatim and reads only the architecture and the per-machine launch command.

### Repo visibility

`--private` / `--public` is tri-state, and **a push never changes visibility on its own**:

| you run | new repo | repo that already exists |
|---|---|---|
| `push repo` (no flag) | created **public** | visibility **left exactly as it is** |
| `push repo --private` | created private | made private, and the change is reported |
| `push repo --public` | created public | made public, and the change is reported |

So re-pushing an update to a private repo cannot publish it by omission, and `--publish` on an
existing private repo asks you for `--public` rather than flipping it for you.

## Kernel-cache bundles (kernels + runner + weights, legacy dispatch path)

> **Legacy.** This path serves through the older dispatch runtime (`tt_api.serve`), not vLLM.
> Prefer a [vLLM bundle](#serving-with-vllm-the-default) for new models.

A kernel-cache bundle ships a precompiled tt-metal cache and can add a runner and a weights
reference so one `pull` installs everything. The runner is either **packaged** (a wheel shipped
in the bundle, via `--python-package`) or a **reference** (a `--runner-spec` the consumer
already has or installs from `--runner-source`). **Producing the runner is governed by
[docs/authoring_runners.md](docs/authoring_runners.md)** — read it before pushing one; a
runner that doesn't follow the contract won't install or serve.

```bash
# Producer (on a host whose kernel cache is populated, with the runner wheel built):
tt-model push you/mymodel-blackhole --private \
  --python-package dist/ttrunner_mymodel-0.1-py3-none-any.whl \
  --runner-spec ttrunner_mymodel.runner:MyRunner \
  --weights some-org/mymodel

# Consumer:
tt-model pull you/mymodel-blackhole       # kernels + pip-install runner + download weights
#   -> prints the exact `serve --unsafe --runner ...` command to run
```

`pull` partial-install flags: `--no-python`, `--no-weights`, `--kernels-only`,
`--models-dir DIR`, `--python PATH` (target interpreter for the runner install).

**Version coupling:** the runner and the kernels are co-versioned (the kernels were compiled
from the tt-metal build whose `ttnn` the runner calls). A kernel-version mismatch hard-blocks;
the runner/weights install anyway with a warning. `tt-model` does not fix a mismatch — build
and serve on the same tt-metal build. See the guide for details.

## The front door: `serve` and `run`

For a vLLM bundle, [`tt-model serve <id>`](#serving-with-vllm-the-default) is the default and
the recommended entry point.

`tt-model run <id>` is the general resolver. It routes a **vLLM bundle to the vLLM plugin**
(the same path as `serve`); anything else falls down the legacy **dispatch three-tier ladder** —
a curated kernel-cache bundle always wins, and a bare Hugging Face repo falls through to the
dynamic dispatch runtime. A completely custom implementation is therefore never overridden.

| Tier | Trigger | What runs |
|------|---------|-----------|
| **vLLM (default)** | a vLLM bundle | the Tenstorrent vLLM plugin (OpenAI server) — see [Serving with vLLM](#serving-with-vllm-the-default) |
| **1 — custom bundle** | a kernel-cache bundle carries a runner | the author's runner + their precompiled kernels (dispatch) |
| **2 — kernels-only** | a kernel-cache bundle with no runner | the dynamic dispatch runtime, with the precompiled cache hitting on disk |
| **3 — no bundle** | a bare HF id / local path | the dynamic dispatch runtime on the model as-is |

```bash
tt-model serve you/mymodel                  # vLLM bundle -> the plugin (default)
tt-model run   you/mymodel-blackhole        # kernel-cache bundle -> author's runner + kernels (dispatch)
tt-model run   meta-llama/Llama-3.1-8B      # no bundle -> dynamic dispatch runtime on the bare repo
tt-model run   you/mymodel --print          # print the serve command instead of executing
tt-model run   you/mymodel --local-only     # resolve only against installed bundles (no Hub call)
```

On the legacy dispatch path, when a tuned bundle is **published but not installed**, `run`
tells you it exists (`tt-model pull <id>` to use it) and then does exactly what you asked —
running the dynamic path on the bare repo rather than silently downloading. That handoff
targets the dispatch runtime (`tt_api.serve`); `tt-model` only *detects* that package, never
imports it — the runner spec is an opaque string.

## Community catalog (web front end)

`web/` is a static, searchable browser for community-published bundles — like
`ollama.com/models`, backed entirely by the Hugging Face Hub. It is a **pure index**: it
hosts and stores nothing, and queries the HF public API live from the visitor's browser.
Every card is a pointer to a public HF repo that remains under its author's governance.

Listing is an explicit opt-in, separate from `push`:

```bash
tt-model push you/mymodel-blackhole --public --publish   # push and list in one step
tt-model publish   you/mymodel-blackhole                  # list a repo pushed earlier
tt-model unpublish you/mymodel-blackhole                  # delist (repo untouched)
```


`--publish` requires `--public` and adds the `tt-model-catalog` tag; the catalog shows only
repos carrying it. Deploy the front end by copying `web/` to any static server — no backend,

no build step. See **[web/README.md](web/README.md)**.

## Checking your toolchain

`tt-model doctor` only ever *checks* — it never installs, so its verdict is always a
report on the machine as it is. Provisioning is the separate, explicit
[`tt-model install`](#install); `doctor`, `instances`, and the compatibility gates stay
declarative.

```bash
tt-model doctor
```

```
Toolchain:
  ✓ tt-metal: 0.72.1.dev3 (require >= 0.72.0) — ok
  ✓ vllm: 0.11.0 (require >= tenstorrent/vllm@dev + plugin) — ok (vllm + TT plugin present)

Hardware:
  ✓ arch=blackhole devices=1 (via tt-smi)
```

The vLLM check is presence-based (the fork tracks the `dev` branch): both `vllm` and the
`vllm_tt_plugin` package must be importable.

`doctor` exits non-zero if any component is missing or below the required version. `run` and
`pull` run the same check and emit a warning (they do not abort) so a version skew is visible
before it bites.

## How compatibility is enforced

A cached binary is only valid when the consumer's environment matches the producer's.
`tt-model` records this in `tt_kernel_manifest.json` and checks it on `pull`:

| Field | Source of truth | On mismatch |
|-------|-----------------|-------------|
| `arch` | tt-smi → `ARCH_NAME` → `--arch` | **fatal** — binaries are a different ISA |
| `tt_metal_version` | package metadata → `git describe` | blocked (use `--force`) — per-kernel hashes won't match |
| `build_key` inputs | tt-smi + env + flags | blocked (use `--force`) — names a different cache dir |
| `device_count` | tt-smi | warning (use `--force`) |

**v4 (vLLM) bundles use version *ranges*, not pins.** A v4 manifest declares
`platform.ttnn` (e.g. `>=0.72,<0.76`), `runtime.version` (vLLM core, e.g. `>=0.24`), and
`runtime.plugin_version` (the `vllm_tt_plugin` package) as PEP 440 specifiers. On `pull`, an
installed version outside a range is a **forceable** block (`--force` overrides), never fatal;
`arch` stays fatal; a bare git-sha checkout is treated as "assume OK". `tt-model doctor <id>`
reports the required-vs-installed verdict declaratively — it never installs (that is
`tt-model install`).

**Multiple tt-metal builds → the instance registry.** When several tt-metal builds are on a
host, `pull` selects the **newest installed instance that satisfies the manifest's ranges** and
pins its activation (interpreter + `TT_METAL_HOME`/`PYTHONPATH`/`LD_LIBRARY_PATH`); `serve`
launches under that exact build. Instances come from the active interpreter, a manager-owned
registry (`~/.config/tt-model/instances.json`), and an auto-scan — manage them with
`tt-model instances list|add|remove|scan` and override per-command with `--instance`. See
**[docs/authoring_runners.md](docs/authoring_runners.md)**.

`build_key` (which names the on-disk cache subtree, `<cache_root>/<build_key>/`) is
computed in C++ and not exposed to Python, so `pull` reconstructs its **inputs** —
`arch`, dispatch core type/axis, `num_hw_cqs`, `harvesting_mask` (only when coordinate
virtualization is disabled), and a compile-flag fingerprint — and refuses to install on
a mismatch. Pass `--probe` to open a device and read the true local `build_key` for an
exact integer check.

These rules mirror the verified tt-metal source: cache root in `rtoptions.cpp` /
`build.cpp`, layout in `jit_compile_server.cpp`, `build_key` in `build_env_manager.cpp`,
and the per-kernel hash in `program_descriptors.cpp`.

## Cache location

Resolved exactly as tt-metal does: `TT_METAL_CACHE` → `$HOME/.cache/tt-metal-cache/` →
`/tmp/tt-metal-cache/`. Override with `--cache-dir`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## Testing without hardware

You don't need a Tenstorrent card or a tt-metal build to exercise the full
push/pull round-trip — generate a synthetic cache and stamp the version by hand:

```bash
# 1. Make fake cache data laid out like a real tt-metal cache
scripts/make_test_cache.sh /tmp/ttk-test-cache 4242

# 2. Auth (use your own current HF token)
export HF_TOKEN=hf_...
tt-model login

# 3. Publish it — --arch and --tt-metal-version stand in for hardware/build detection
tt-model push <you>/kernel-selftest --private \
  --cache-dir /tmp/ttk-test-cache --arch blackhole \
  --tt-metal-version v0.99-test

# 4. Inspect + compatibility verdict
tt-model info <you>/kernel-selftest --arch blackhole

# 5. Pull into a DIFFERENT empty cache dir (simulates another machine)
tt-model pull <you>/kernel-selftest --cache-dir /tmp/ttk-restore --arch blackhole
diff -r /tmp/ttk-test-cache/tt-metal-cache4242 /tmp/ttk-restore/tt-metal-cache4242 \
  && echo "round-trip OK"

# 6. Local bookkeeping + teardown
tt-model list
tt-model rm <you>/kernel-selftest --cache-dir /tmp/ttk-restore
```

Try the guard rails too: `tt-model pull ... --arch wormhole_b0` fails fatally
(wrong ISA), and a `--tt-metal-version` that differs from the bundle's blocks the
install until you add `--force`.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs via GitHub Issues
- Submitting pull requests
- Coding standards and testing requirements

Pull requests are reviewed weekly. For questions, feel free to open an issue or discussion.

## License

This project is licensed under the **Apache License 2.0** - see [LICENSE](LICENSE) for the complete license text.

For clarification on how this license applies to commercial use, modifications, and patent grants, see [LICENSE_understanding.txt](LICENSE_understanding.txt).

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.
