# `tt-model`

> ⚠️ **Experimental. No support, no guarantees.** `tt-model` is an early, experimental
> project. Nothing here is officially supported, and Tenstorrent makes no claim of
> correctness, stability, or fitness for any purpose. APIs, the bundle format, and behavior
> may change or break at any time without notice. Use it at your own risk.

`tt-model` publishes and pulls **self-contained model bundles** over the Hugging Face (HF) Hub
and serves them on Tenstorrent hardware through the
[Tenstorrent vLLM plugin](https://github.com/tenstorrent/vllm-tt-plugin), an OpenAI-compatible
server. Every bundle ships or builds its **own** serving stack in a per-model virtual
environment (venv) or container image, so the host needs only a Tenstorrent PCIe card and its
firmware. A bundle does not contain model weights. It records the weights' HF repo and fetches
them on pull or on first serve.

## Install

`tt-model` is a normal Python package. It is not on PyPI, so install it from a clone:

```bash
git clone https://github.com/tenstorrent/tt-model-manager && cd tt-model-manager
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

There is nothing else to provision on the host. Any `ttnn` or vLLM already installed on the host
is not required and is not touched.

## Quickstart

```bash
tt-model login                       # reuses huggingface_hub's token store
tt-model search gemma                # discover published bundles
tt-model serve <org>/<model>         # pull and install if needed, then launch the OpenAI server
tt-model curl "hello"                # send a chat completion to the running model
```

New here? There is a copy-paste, end-to-end recipe per package format:
[docs/E2E_RECIPE_V5.1.md](docs/E2E_RECIPE_V5.1.md) for a v5.1 container package (the
supported path) and [docs/E2E_RECIPE_V6.md](docs/E2E_RECIPE_V6.md) for a v6 thin bundle. Both
walk through the same five steps in order:

1. Package a model you have brought up.
2. Push the bundle to the HF Hub.
3. Pull it on a consumer host.
4. Serve it.
5. Verify it answers.

## Package formats

A bundle is an HF **model** repo that carries or pins the whole serving stack. There are two
kinds. Both are served with the same `tt-model serve` command.

- **v5.1 container**. Ships the whole platform as an Open Container Initiative (OCI) image. The
  consumer needs Docker and a Tenstorrent PCIe card. Authored from a single `tt-model.yaml` with
  `tt-model package --container`.
  Reference: [docs/container_packages.md](docs/container_packages.md).
- **v6 thin** (**beta, unsupported**). Builds the venv from pip pins plus bundled wheels.
  Author it with `tt-model package-thin`.
  Reference: [docs/thin_packages.md](docs/thin_packages.md).

`tt-model` refuses bundles published with an older schema. Re-publish them with a current `tt-model`.

## Documentation

| Page | What it covers |
|---|---|
| [docs/E2E_RECIPE_V5.1.md](docs/E2E_RECIPE_V5.1.md) | Step-by-step for a v5.1 container package: bring-up, `tt-model.yaml`, package, push, pull, serve, verify, with troubleshooting |
| [docs/E2E_RECIPE_V6.md](docs/E2E_RECIPE_V6.md) | Step-by-step for a v6 thin bundle: bring-up, package-thin, push, pull, serve, verify, with troubleshooting |
| [docs/cli.md](docs/cli.md) | Every `tt-model` command with its common flags |
| [docs/publishing.md](docs/publishing.md) | push vs. public vs. publish, the community catalog, how compatibility is checked |
| [docs/container_packages.md](docs/container_packages.md) | v5.1 container packages: `tt-model.yaml`, serving, how the container runs, and design notes |
| [docs/thin_packages.md](docs/thin_packages.md) | v6 thin bundles (draft) |
| [docs/multi_host_mesh.md](docs/multi_host_mesh.md) | Multi-host (cross-box) mesh design note (issue #87) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, testing, PR process |
| [AGENTS.md](AGENTS.md) | Design invariants and workflow notes for maintainers and coding agents |

## Development

The offline test suite needs no Tenstorrent card, `ttnn`, vLLM, or Hugging Face token. With
[`uv`](https://docs.astral.sh/uv/), use the checked-in lockfile:

```bash
uv sync --locked --extra test
uv run --locked --extra test pytest
```

For the plain `venv` and pip path, see [Development setup](CONTRIBUTING.md#development-setup).

## Related tooling

If you are still bringing a model up on TT-Metalium™ before packaging it,
[`tenstorrent/skills`](https://github.com/tenstorrent/skills) is a separate Claude Code and Codex
plugin marketplace for TT-Metalium bring-up, review, and debugging. `tt-model` packages,
publishes, and serves models that already work.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for reporting bugs,
submitting pull requests, and coding standards. Pull requests are reviewed weekly.

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE), and
[LICENSE_understanding.txt](LICENSE_understanding.txt) for how it applies to commercial use,
modifications, and patent grants. This project follows the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
