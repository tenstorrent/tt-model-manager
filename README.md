# tt-model

> ⚠️ **Experimental — no support, no guarantees.** tt-model is an early, experimental
> project. Nothing here is officially supported, and we make no claim of correctness,
> stability, or fitness for any purpose. APIs, the bundle format, and behavior may change
> or break at any time without notice. Use it at your own risk.

`tt-model` publishes and pulls **self-contained model bundles** over the Hugging Face Hub and
serves them on Tenstorrent hardware through the **Tenstorrent vLLM plugin**
([`tenstorrent/vllm-tt-plugin`](https://github.com/tenstorrent/vllm-tt-plugin) — stock upstream
vLLM built `VLLM_TARGET_DEVICE=empty`), an OpenAI-compatible server. Every bundle carries or
builds its **own per-model venv**, so the box needs only a TT card + firmware (plus SFPI, an
externally-managed box dependency). One command pulls a bundle and brings the server up:

```bash
tt-model serve <namespace>/<model>     # install if needed, then launch the OpenAI server
```

## Bundles

A bundle is a self-contained HF **model** repo: it ships (or pins) the whole serving stack and
builds it into a fresh, per-model venv on the consumer. Weights are **never embedded** — only
referenced by their HF repo id and fetched at pull or first load. Both bundle kinds render the
plugin-owned `vllm_metadata.json` (the `EXTRA_MODELS_DIR` contract: HF arch name → main_class)
and both serve the same way. There are exactly two schemas:

- **v5 "fat"** (schema_version `5`, the `bundled` block) embeds the author's built artifacts —
  their `ttnn` wheel (custom kernels compiled in), an empty-target vLLM wheel, the vLLM plugin
  wheel, and their modified `tt-metal-community` tree — which the bundle's `install.sh` installs
  into a fresh venv. Author it with **`tt-model package`**. See
  **[docs/self_contained_packages.md](docs/self_contained_packages.md)**.
- **v6 "thin"** (schema_version `6`, the `deps` block) builds the venv from pip dependency pins
  (`ttnn` / `tt-metal-models`) plus bundled wheels (the `vllm-tt-plugin` and any `generic_op`
  custom-op wheel) plus an empty-target vLLM build step. No embedded `ttnn` wheel, no metal
  tree. Author it with **`tt-model package-thin`**. See
  **[docs/thin_packages.md](docs/thin_packages.md)**.

Older bundles (pre-v5 schemas) are refused: *re-publish the bundle with a current tt-model.*

## How models are packaged (v5 self-contained — recommended)

> **New here?** The copy-paste, end-to-end recipe is
> **[docs/E2E_RECIPE.md](docs/E2E_RECIPE.md)**: model → package → push → pull → serve.

The recommended format is a **self-contained (v5) bundle** — *"package what's on your box."* One
Hugging Face **model** repo carries everything needed to run, so a consumer needs **only a TT card
+ firmware** — no tt-cli, no pre-provisioned tt-metal, no host vLLM:

```
wheels/            the author's built ttnn wheel (custom C++/LLK kernels compiled in),
                   base vLLM + plugin wheels, and the vendored dependency closure   (git-LFS)
metal/             the author's modified tt-metal-community tree (Python blocks + model code)
vllm_models/<name>/vllm_metadata.json     the plugin's EXTRA_MODELS_DIR contract
install.sh  run.sh  requirements.txt      generated launcher + installer
tt_kernel_manifest.json                   the v5 manifest
# weights: NOT embedded — a pointer (HF repo id) in the manifest, fetched on pull/serve
```

**Why this shape:**

- **Self-contained.** The engine that runs is the author's actual build (their
  kernels ride along inside the `ttnn` wheel), not a stock pin — so custom C++/LLK kernels just work.
- **The folder is a hard wall.** After `pull`, everything needed to serve lives *under the install
  directory* — the pinned interpreter (provisioned by `uv` into the folder), the venv, the engine,
  the model code, and, on first serve, the weights and all caches. Serving depends on nothing
  outside the folder except the TT device and system libc. Only `pull` touches the network.
- **Portable across machines.** The `ttnn` wheel is made portable with `auditwheel` (vendored libs +
  `$ORIGIN` RPATH); deps are vendored so install is offline + reproducible; `pull` verifies the
  wheel's interpreter/arch/**glibc** floor and fails clearly on a mismatch instead of at runtime.
  Build the engine wheel on **Ubuntu 22.04 (glibc 2.35)** and one bundle runs on both 22.04 and 24.04.
- **Weights stay a pointer** — they're large and shared, so the manifest references the HF repo id.

```bash
# Producer — capture what's on your box and publish
tt-model package <org>/<model> --from-metal . --wheels-dir ./wheels \
  --arch blackhole --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct --mesh P150 --vendor-deps --repair

# Consumer — only a card + firmware needed
tt-model pull  <org>/<model>
tt-model serve <org>/<model>
```

Full details: **[docs/self_contained_packages.md](docs/self_contained_packages.md)**.

## Install

`tt-model` itself is a normal Python package:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

There is nothing to provision on the box beyond a TT card + firmware (and SFPI). Each bundle
builds its own venv from what it ships or pins, so the host's `ttnn`/vLLM (if any) is never
required and never touched.

## Usage

```bash
tt-model login                                    # reuses huggingface_hub's token store

# Run a model — serve a bundle through the Tenstorrent vLLM plugin
tt-model serve you/mymodel                         # install if needed, then launch the OpenAI server
tt-model serve you/mymodel --print                 # print the exact launch command instead of running it
tt-model serve you/mymodel --local-only            # require an installed bundle; never hit the Hub

# Get models
tt-model pull   you/mymodel                         # download + install the bundle into its own venv
tt-model pull   you/mymodel --with-weights          # ...and pre-download the weights (default: skip)
tt-model info   you/mymodel                         # manifest + compatibility verdict
tt-model search gemma                               # discover published bundles
tt-model search gemma --catalog                     # only bundles listed in the community catalog
tt-model search --arch blackhole                    # only bundles tagged for an arch
tt-model list                                       # locally installed bundles
tt-model rm     you/mymodel                         # remove an installed bundle

# Publish models
tt-model package      you/mymodel ...               # author + push a v5 fat bundle
tt-model package-thin you/mymodel ...               # author + push a v6 thin bundle
tt-model publish      you/mymodel                    # list a public bundle in the community catalog
tt-model unpublish    you/mymodel                    # delist (repo untouched)

tt-model version                                    # print the installed tt-model version
```

## Serving

`tt-model serve <id>` is the one-command path. For an already-installed bundle it runs the
bundle's `run.sh` directly from that bundle's own venv — the host toolchain is irrelevant
because the bundle ships or builds its own. For a bundle that isn't installed yet, `serve`
downloads it, runs its `install.sh` to build the per-model venv, then serves. `run.sh` wires the
engine env and launches the OpenAI-compatible vLLM server.

```bash
tt-model serve you/mymodel                    # install-if-needed → launch; prints the endpoint
tt-model serve you/mymodel --print            # emit the exact launch command + env instead of running
tt-model serve you/mymodel --local-only       # require an installed bundle; never hit the Hub
tt-model serve you/mymodel -- --extra vllm-arg # anything after the id is passed through to vLLM
```

Repeat invocations skip the install and go straight to launch.

### Checking it answers

`tt-model curl` builds the chat-completions request for whatever is being served, so
verifying a bring-up doesn't mean hand-writing JSON and matching the model id exactly:

```bash
tt-model curl "hello"                        # send it to the running model
tt-model curl "write a haiku" --temperature 0.7 --max-tokens 200
tt-model curl "hello" --print                # emit the equivalent curl instead of sending
```

The model id comes from the running server (`GET /v1/models`); with nothing serving yet,
`--print` falls back to the installed bundle's weights id so it still emits something
pasteable. Any option the command doesn't reserve (`--print`, `--model`, `--base-url`) goes
straight into the request body, so the whole vLLM sampling surface is available.

## Publishing a bundle

Author on the box where you built/brought up the model, then push. The full authoring
recipe — every flag, the resulting repo layout, and the offline + hardware tests — lives in the
per-schema guides:

```bash
# v5 fat: embeds your built ttnn wheel + vLLM/plugin wheels + your tt-metal-community tree
tt-model package you/mymodel --public \
  --from-metal ./tt-metal-community \
  --ttnn-wheel dist/ttnn-*.whl \
  --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct         # POINTER — weights are not embedded

# v6 thin: builds the venv from pip pins (ttnn / tt-metal-models) + bundled wheels
tt-model package-thin you/mymodel --public ...

tt-model pull you/mymodel                          # lay the bundle in + build its venv
tt-model pull you/mymodel --with-weights           # ...and also pre-download the weights
```

`--out <dir>` stages the running folder locally without pushing. Large wheels go to git-LFS
automatically on push. See **[docs/self_contained_packages.md](docs/self_contained_packages.md)**
(v5) and **[docs/thin_packages.md](docs/thin_packages.md)** (v6).

### Repo visibility

`package` / `package-thin` default to `--private`; pass `--public` to publish openly. A push
never lists your bundle in the community catalog on its own — that is a separate opt-in
(`--publish`, which requires `--public`, or `tt-model publish` later).

## Community catalog

Published bundles can opt into a searchable community catalog of community-published models.
The catalog — its web front end and its indexer — lives in a dedicated repo,
**[tenstorrent/model-manager-site](https://github.com/tenstorrent/model-manager-site)**
(formerly the `web/` directory here). Every listing is a pointer to a public HF repo that
remains under its author's governance.

Listing is an explicit opt-in, separate from the push:

```bash
tt-model package   you/mymodel --public --publish   # push and list in one step
tt-model publish   you/mymodel                       # list a repo pushed earlier
tt-model unpublish you/mymodel                        # delist (repo untouched)
```

`--publish` requires `--public` and adds the `tt-model-catalog` tag
(`TT_MODEL_CATALOG_TAG` in [`tt_kernel/__init__.py`](src/tt_kernel/__init__.py)); the catalog
indexes only repos carrying it, and reads each repo's `tt_kernel_manifest.json` to render it.

## How compatibility is checked

`tt-model` records the target arch/machine in the bundle's `tt_kernel_manifest.json` and reports
a verdict on `pull`/`serve`/`info` (`compare()` in `compat.py`). `arch` mismatch is fatal —
binaries are a different ISA; other mismatches are non-fatal and overridable with `--force`.
Because each bundle builds its own venv, there is no host `ttnn`/vLLM version to gate against.
`tt-model info <id>` prints the manifest and the required-vs-detected verdict declaratively.

## Development

The development environment is intentionally smaller than the serving environment: the
offline test suite does not require a TT card, `ttnn`, vLLM, or a Hugging Face token. With
[`uv`](https://docs.astral.sh/uv/), use the checked-in lockfile:

```bash
uv sync --locked --extra test
uv run --locked --extra test pytest
```

For the standard-library `venv` + pip path and more detail, see
[Development setup](CONTRIBUTING.md#development-setup).

<details>
<summary>Setup without uv</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

</details>

The producer/consumer logic is fully unit-tested with mocked pip + HF, so you can exercise the
full package → pull → serve round-trip without a card. See the Testing sections of
[docs/self_contained_packages.md](docs/self_contained_packages.md) and
[docs/thin_packages.md](docs/thin_packages.md).

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
