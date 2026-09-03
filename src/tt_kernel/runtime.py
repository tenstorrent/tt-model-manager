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
from typing import List, Optional

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


# ------------------------------------------------------------------ verify (tt-model curl)
# The consumer's last step is "did it actually answer?". Everything below builds that one
# OpenAI chat request so the user never has to retype the model id, the endpoint or the JSON
# body. Stdlib only — tt-model takes no HTTP dependency.
# Tracks manifest.DEFAULT_PORT, where `tt-model serve` puts a server when nothing names a
# port. If serve had to walk past a busy 20000 (it says so, and prints the endpoint),
# point curl there with --base-url or TT_MODEL_BASE_URL.
DEFAULT_BASE_URL = "http://localhost:20000"
ENV_BASE_URL = "TT_MODEL_BASE_URL"
DEFAULT_PROMPT = "Say hello in one sentence."
DEFAULT_MAX_TOKENS = 64


def list_models(base_url: str, *, timeout: float = 5.0) -> List[str]:
    """The model ids an OpenAI-compatible server currently serves (``GET /v1/models``).

    This is the authoritative answer to "what is running": it is the id vLLM registered, not
    something we recorded at install time and hope is still true. Returns ``[]`` when the
    server is unreachable or answers something unexpected — the caller falls back to the
    install record, because printing a command must work with nothing listening.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — localhost probe
            payload = _json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return [m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


def parse_param(raw: str) -> object:
    """Coerce a CLI-supplied sampling value to its JSON type, falling back to the string.

    ``0.7`` -> float, ``200`` -> int, ``true`` -> bool, a JSON array -> list. A bare word
    like ``length`` isn't valid JSON, so it stays a string — which is what the API wants.
    """
    import json as _json

    try:
        return _json.loads(raw)
    except ValueError:
        return raw


def parse_extra_params(extra: List[str]) -> dict:
    """Fold leftover ``--key value`` CLI args into OpenAI request fields.

    vLLM's sampling surface moves faster than this CLI should; rather than enumerate
    ``temperature``/``top_p``/``seed``/... as flags that go stale, anything the command
    doesn't reserve is passed through. ``--top-p 0.9`` becomes ``{"top_p": 0.9}`` (dashes to
    underscores, value typed by :func:`parse_param`); a lone ``--flag`` becomes ``True``.

    Raises ``ValueError`` on a bare positional, which is almost always a quoting mistake
    (``tt-model curl hello there``) and would otherwise be silently dropped.
    """
    params: dict = {}
    i = 0
    while i < len(extra):
        token = extra[i]
        if not token.startswith("--"):
            raise ValueError(f"unexpected argument {token!r} (quote the prompt, or use --key value)")
        key, sep, inline = token[2:].partition("=")
        if sep:
            value: object = parse_param(inline)
        elif i + 1 < len(extra) and not extra[i + 1].startswith("--"):
            value = parse_param(extra[i + 1])
            i += 1
        else:
            value = True
        params[key.replace("-", "_")] = value
        i += 1
    return params


def chat_payload(model: str, prompt: str, *, params: Optional[dict] = None) -> dict:
    """The ``/v1/chat/completions`` body for a one-shot prompt.

    ``max_tokens`` is a default rather than a fixed field, so ``--max-tokens 200`` (or any
    other sampling param) simply overrides it.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    body.update(params or {})
    return body


def curl_argv(base_url: str, payload: dict) -> List[str]:
    """The real ``curl`` argv for a chat request — what the command runs verbatim.

    ``-sS``: no progress meter, but transport errors still surface (bare ``-s`` would make a
    refused connection look like an empty reply).
    """
    import json as _json

    return [
        "curl", "-sS", base_url.rstrip("/") + "/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", _json.dumps(payload),
    ]


def render_curl(argv: List[str]) -> str:
    """``argv`` as a copy-pasteable multi-line shell command.

    Shaped like the recipe doc's snippet (one flag pair per continued line) and quoted with
    ``shlex.quote``, so pasting it into a shell sends exactly what the command sends.
    """
    import shlex

    head = " ".join(shlex.quote(a) for a in argv[:3])
    lines = [head]
    rest = argv[3:]
    for i in range(0, len(rest), 2):
        lines.append("  " + " ".join(shlex.quote(a) for a in rest[i:i + 2]))
    return " \\\n".join(lines)


__all__ = [
    "ENV_MODELS_DIR",
    "resolve_models_dir",
    "download_weights",
    "install_self_contained",
    "DEFAULT_BASE_URL",
    "ENV_BASE_URL",
    "DEFAULT_PROMPT",
    "DEFAULT_MAX_TOKENS",
    "list_models",
    "parse_param",
    "parse_extra_params",
    "chat_payload",
    "curl_argv",
    "render_curl",
]
