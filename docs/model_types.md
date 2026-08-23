# Model types

`type:` in a tt-model manifest selects how the serving environment is built and how the
container is launched. This file is the human-readable half of the enumeration; the code
half — the single source of truth the error messages read from — is
`src/tt_model/types/__init__.py`. Keep the two in step.

## The types

| `type` | vLLM comes from | TT platform plugin comes from | launched with |
| --- | --- | --- | --- |
| `vllm` | stock `vllm==<runtime.vllm.version>` built from sdist with `VLLM_TARGET_DEVICE=empty` against CPU torch | the standalone public repo `tenstorrent/vllm-tt-plugin` at `runtime.plugin.{repo, ref}`, installed non-editable | `vllm serve <weights>` |
| `vllm-legacy` | the `tenstorrent/vllm` **fork** at `runtime.vllm.{repo, ref}`, `VLLM_TARGET_DEVICE=empty`, installed *editable* | the fork's own in-tree `plugins/vllm-tt-plugin` | `python -m models.common.readiness_check.run_vllm_server --stages serve` |

### `vllm`

```yaml
type: vllm
runtime:
  vllm: {version: "0.24.0"}
  plugin: {repo: https://github.com/tenstorrent/vllm-tt-plugin, ref: main}
  extension: models/autoports/<name>/vllm_ext     # optional
  lock: requirements.lock                          # optional but recommended
```

The published vLLM wheel is the CUDA build, so vLLM always builds from sdist with the
`empty` device target; the TT platform is provided by the plugin at runtime (it
activates only when `ttnn` is importable). The plugin installs **non-editable** so its
clone does not have to survive into the runtime image.

Two facts of life this type handles for you:

- ttnn pins `numpy<2` while recent vLLM's `opencv-python-headless` wants `numpy>=2` — a
  hard conflict resolved with a uv `--override` (numpy<2 wins; opencv 4.11 is the last
  release without a numpy-2 floor, and vLLM only reaches opencv through a video-IO path
  no TT model uses). Extra overrides: `runtime.overrides: [...]`.
- `torchaudio` is uninstalled after the vLLM install: transformers imports it if it is
  merely present, and the wheel riding along with CPU torch is unloadable.

### `vllm-legacy`

```yaml
type: vllm-legacy
runtime:
  vllm: {repo: https://github.com/tenstorrent/vllm, ref: bf98d556}   # plugin is in-tree
  model_dir: models/autoports/<name>            # required: the launcher takes --model-dir
  lock: requirements.lock
```

For models whose serving recipe is the tt-metal readiness runner — it resolves the
plugin's config flag against the installed vLLM, forwards an explicit mesh grid, and
hands off to a stock `vllm.entrypoints.openai.api_server`. These models expect the
runner's flag names (`--tt-config`, `--additional-server-args`, `--server-timeout`), so
the launcher is a property of the type, not a third manifest axis.

Both the fork and its in-tree plugin are installed **editable** (the fork's documented
install), so the fork checkout survives into the runtime image at `/opt/vllm` — about
200 MB extra, which is part of why this type is called legacy.

## Choosing

- New model against a released vLLM + the public plugin → `vllm`.
- Model bring-up living on the `tenstorrent/vllm` fork and launched through
  `run_vllm_server` → `vllm-legacy`.

## Adding a type

A future type (diffusion, CNN, another engine) is:

1. a new module in `src/tt_model/types/` implementing the `ModelType` protocol
   (`types/base.py`): validate / install_lines / verify_lines / runtime_copy_lines /
   serve_argv / serve_env / ready_probe;
2. a registration line in `src/tt_model/types/__init__.py`;
3. a row in this file.

No changes to the Dockerfile (type-specific work arrives via the generated
`install_engine.sh` / `verify.sh`), the manifest schema, or any CLI command.
