# Contributing to tt-model

We welcome contributions to tt-model! This document provides guidelines for contributing to the project.

> **Automated / agent contributors (e.g. Claude Code):** read [AGENTS.md](AGENTS.md) for the binding design invariants, testing requirements, and PR discipline before opening a fix.

## Reporting Bugs

If you discover a bug, please report it via [GitHub Issues](https://github.com/tenstorrent/tt-model-manager/issues).

When reporting a bug, please include:
- A clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, tt-metal version, hardware)
- Any relevant logs or error messages

## Submitting Pull Requests

We accept bug fixes and new functionality through Pull Requests (PRs).

### Before You Submit

1. **Search existing issues and PRs** to avoid duplicates
2. **Discuss significant changes** by opening an issue first
3. **Follow the project's coding standards** and conventions
4. **Write tests** for new functionality
5. **Update documentation** as needed

### PR Process

1. Fork the repository and create a new branch from `main`
2. Make your changes in the branch
3. Run tests to ensure they pass: `pytest`
4. Commit your changes with clear, descriptive commit messages
5. Push your branch and submit a PR

### Review Process

- PRs are reviewed weekly
- Maintainers will provide feedback or approve your changes
- Address any requested changes
- Once approved, a maintainer will merge your PR

## Coding Standards

- Follow PEP 8 style guidelines for Python code
- Use type hints where appropriate
- Write clear, concise docstrings for functions and classes
- Keep functions focused and modular

## Development setup

`tt-model` has two deliberately different environments:

- A **development environment** for editing the package and running its offline tests. It
  does not need Tenstorrent hardware, `ttnn`, vLLM, or Hugging Face credentials.
- A **serving environment** provisioned per model by a self-contained bundle's own
  `install.sh` (v5 fat / v6 thin). That path builds the bundle's own venv, expects `ttnn`,
  installs the Tenstorrent vLLM stack, and is not needed for ordinary development.

Python 3.9 or newer is required. The reproducible setup uses
[`uv`](https://docs.astral.sh/uv/) and the checked-in lockfile:

```bash
uv sync --locked --extra test
uv run --locked --extra test pytest
```

`uv` creates and uses this checkout's `.venv`. If another virtual environment is active,
deactivate it first so `uv` does not warn about the mismatch; do not add `--active` unless
you intentionally want to modify that other environment.

Without `uv`, create the same isolated editable install with the standard library and pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

When project dependencies change, regenerate and verify the committed lockfile with
`uv lock` and `uv lock --check`.

## Testing

All code changes should include appropriate tests:
- Unit tests for new functions and classes
- Integration tests for end-to-end workflows
- Run the full offline suite with: `python -m pytest`

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion in GitHub Issues
- Review existing documentation in the [docs/](docs/) directory

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache 2.0 License.
