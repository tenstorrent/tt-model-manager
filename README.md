# tt-model

> ⚠️ **Experimental — no support, no guarantees.** tt-model is an early, experimental
> project. Nothing here is officially supported, and we make no claim of correctness,
> stability, or fitness for any purpose. APIs, the bundle format, and behavior may change
> or break at any time without notice. Use it at your own risk.

`tt-model` publishes and pulls **self-contained model bundles** over the Hugging Face Hub and
serves them on Tenstorrent hardware through the
[Tenstorrent vLLM plugin](https://github.com/tenstorrent/vllm-tt-plugin), an OpenAI-compatible
server. Every bundle ships or builds its **own** serving stack in its own per-model venv (or
container image), so the box needs only a TT card + firmware. Weights are never embedded — a
bundle points at their HF repo and fetches them on pull or first serve.

## Install

`tt-model` itself is a normal Python package:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

There is nothing else to provision on the box. The host's `ttnn`/vLLM (if any) is never
required and never touched.

## Quickstart

```bash
tt-model login                       # reuses huggingface_hub's token store
tt-model search gemma                # discover published bundles
tt-model serve <org>/<model>         # pull + install if needed, then launch the OpenAI server
tt-model curl "hello"                # send a chat completion to the running model
```

> **New here?** The copy-paste, end-to-end recipe is
> **[docs/E2E_RECIPE.md](docs/E2E_RECIPE.md)**: model → package → push → pull → serve.

## Package formats

A bundle is an HF **model** repo that carries or pins the whole serving stack. There are three
kinds, and all serve the same way:

- **v5 self-contained** (recommended) — embeds the author's built `ttnn` wheel (custom kernels
  compiled in), vLLM + plugin wheels, and their `tt-metal-community` tree; the consumer builds a
  venv from them at `pull`. Author with `tt-model package`.
  → [docs/self_contained_packages.md](docs/self_contained_packages.md)
- **v5.1 container** — ships the whole platform as an OCI image; the consumer needs Docker + a
  TT card. Authored from a single `tt-model.yaml` with `tt-model package --container`.
  → [docs/container_packages.md](docs/container_packages.md)
- **v6 thin** (**beta, unsupported**) — builds the venv from pip pins plus bundled wheels.
  Author with `tt-model package-thin`. → [docs/thin_packages.md](docs/thin_packages.md)

Pre-v5 bundles are refused; re-publish them with a current `tt-model`.

## Documentation

| Page | What it covers |
|---|---|
| [docs/E2E_RECIPE.md](docs/E2E_RECIPE.md) | Step-by-step: bring-up → package → push → pull → serve → verify, with troubleshooting |
| [docs/cli.md](docs/cli.md) | Every `tt-model` command with its common flags |
| [docs/publishing.md](docs/publishing.md) | push vs. public vs. publish, the community catalog, how compatibility is checked |
| [docs/self_contained_packages.md](docs/self_contained_packages.md) | v5 design, layout, and testing reference |
| [docs/container_packages.md](docs/container_packages.md) | v5.1 container packages: `tt-model.yaml`, the image on the wire, serving |
| [docs/thin_packages.md](docs/thin_packages.md) | v6 thin bundles (draft) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, testing, PR process |
| [AGENTS.md](AGENTS.md) | Design invariants and workflow notes for maintainers and coding agents |

## Development

The offline test suite needs no TT card, `ttnn`, vLLM, or Hugging Face token. With
[`uv`](https://docs.astral.sh/uv/), use the checked-in lockfile:

```bash
uv sync --locked --extra test
uv run --locked --extra test pytest
```

For the plain `venv` + pip path, see [Development setup](CONTRIBUTING.md#development-setup).

## Related tooling

Bringing a model up on tt-metal before packaging it? [`tenstorrent/skills`](https://github.com/tenstorrent/skills)
is a separate Claude Code / Codex plugin marketplace for tt-metal bring-up, review, and
debugging. `tt-model` packages, publishes, and serves models that are already working.

## Contributing

We welcome contributions — see [CONTRIBUTING.md](CONTRIBUTING.md) for reporting bugs,
submitting pull requests, and coding standards. Pull requests are reviewed weekly.

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE), and
[LICENSE_understanding.txt](LICENSE_understanding.txt) for how it applies to commercial use,
modifications, and patent grants. This project follows the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
