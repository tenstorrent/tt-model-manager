# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Install the runtime half of a self-contained bundle: the weights and the venv.

A v5/v6 bundle carries its own ``install.sh`` (which builds the venv and installs the engine) and
its own ``run.sh`` (which serves). This module downloads the model weights and drives ``install.sh``;
everything else the bundle does for itself.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from . import compat
from .manifest import WeightsRef

ENV_MODELS_DIR = "TT_MODEL_MODELS_DIR"


def resolve_models_dir(models_dir: Optional[str], repo_id: str) -> Path:
    """Where to download a model's weights / install its bundle.

    Resolution (env-then-flag): ``--models-dir`` > ``TT_MODEL_MODELS_DIR`` >
    ``~/.cache/tt-model/models``. The repo id is nested as ``<base>/<org>/<name>`` (no
    slash-flattening) so the path round-trips cleanly for ``rm``/serve and never collides.
    """
    explicit = models_dir if models_dir is not None else compat.env(ENV_MODELS_DIR)
    if explicit:
        base = Path(explicit).expanduser()
    else:
        home = os.environ.get("HOME")
        base = compat.data_dir(Path(home) / ".cache" if home else Path("/tmp")) / "models"
    # repo_id is "org/name" (or just "name"); keep its structure under base.
    return base.joinpath(*repo_id.split("/"))


def download_weights(weights: WeightsRef, dest: Path) -> Path:
    """Download a model's weights from the Hub into ``dest`` (resumable).

    Thin wrapper over ``huggingface_hub.snapshot_download`` — content-addressed and
    resumable, so a half-finished download just continues on a re-pull.
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=weights.repo_id,
        repo_type=weights.repo_type,
        revision=weights.revision,
        allow_patterns=weights.allow_patterns,
        ignore_patterns=weights.ignore_patterns,
        local_dir=str(dest),
    )
    return Path(path)


def install_self_contained(bundle_dir: Path, venv_dir: Path) -> Path:
    """Run a self-contained bundle's ``install.sh`` to build its own venv.

    The generated ``install.sh`` creates ``venv_dir`` and installs the engine — for v5, the shipped
    wheels (the author's ttnn + empty-target vLLM + plugin); for v6, ttnn/tt-metal-models from the
    index plus the empty-target vLLM build and the plugin/ops wheels. Returns the venv's python.
    Raises CalledProcessError on failure. This is the "install the platform" step that makes a
    package need only a card + firmware.
    """
    script = bundle_dir / "install.sh"
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found (not a self-contained bundle).")
    subprocess.run(["bash", str(script), str(venv_dir)], check=True)
    return venv_dir / "bin" / "python"


__all__ = [
    "ENV_MODELS_DIR",
    "resolve_models_dir",
    "download_weights",
    "install_self_contained",
]
