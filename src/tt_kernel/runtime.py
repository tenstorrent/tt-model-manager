# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Install the *runtime* half of a bundle: the Python runner wheel and the weights.

This module deliberately holds everything that is NOT kernel-cache plumbing (cache.py)
or bundle-repo I/O (hub.py): downloading an arbitrary HF *model* repo, pip-installing
the shipped runner wheel into the active venv, and composing the ready-to-run serve
command. It NEVER imports the dispatch serving package — the runner spec is an opaque
string and dispatch is only *detected*, never imported (the decoupling boundary).
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

from . import compat
from typing import List, Optional

from .manifest import WeightsRef

ENV_MODELS_DIR = "TT_MODEL_MODELS_DIR"
# tt-model's own minimal OpenAI server for a legacy (dispatch-contract) runner. This
# replaces the retired dispatch runtime the runner used to hand off to. Used to BUILD the
# serve command; the module is only run as a subprocess, never imported here.
_LEGACY_SERVE_MODULE = "tt_kernel.legacy_serve"
# The env var the Tenstorrent vLLM plugin reads to discover extra model bundle folders.
# tt-model points it at the local bundles_dir at serve time; the plugin scans it and
# registers every model folder found there.
ENV_EXTRA_MODELS_DIR = "EXTRA_MODELS_DIR"
_VLLM_PKG = "vllm"
_VLLM_PLUGIN_PKG = "vllm_tt_plugin"


def resolve_models_dir(models_dir: Optional[str], repo_id: str) -> Path:
    """Where to download a model's weights.

    Resolution (env-then-flag, mirroring cache.resolve_out_root): ``--models-dir`` >
    ``TT_MODEL_MODELS_DIR`` > ``~/.cache/tt-model/models``. The repo id is nested as
    ``<base>/<org>/<name>`` (no slash-flattening) so the path round-trips cleanly for
    ``rm``/serve and never collides.
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


def pip_install_wheels(
    wheel_paths: List[Path],
    *,
    python: Optional[str] = None,
    pip_args: Optional[List[str]] = None,
) -> None:
    """pip-install the shipped runner wheel(s) into the target interpreter's env.

    ``--no-deps`` is deliberate: the runner wheel is tree-shaken/self-contained, and we
    must NOT let pip pull a conflicting ``ttnn`` from PyPI (ttnn/tt-metal is the platform
    the version warning points at, never a vendored dep). ``python`` overrides the target
    interpreter (default: the venv tt-model itself runs in, where ttnn should live);
    ``pip_args`` is an escape hatch for the rare case the wheel really needs extra flags.
    Raises CalledProcessError on a non-zero pip exit.
    """
    if not wheel_paths:
        return
    exe = python or sys.executable
    cmd = [exe, "-m", "pip", "install", "--no-deps"]
    if pip_args:
        cmd.extend(pip_args)
    cmd.extend(str(p) for p in wheel_paths)
    subprocess.run(cmd, check=True)


def ttnn_importable(python: Optional[str] = None) -> bool:
    """Whether ``ttnn`` is importable from the target interpreter.

    Used to warn when pip would install the runner into a venv that lacks ttnn (e.g.
    tt-model was installed via pipx into its own env). For the default interpreter we
    check this process directly; for an explicit ``--python`` we shell out.
    """
    if python is None or python == sys.executable:
        return importlib.util.find_spec("ttnn") is not None
    try:
        proc = subprocess.run(
            [python, "-c", "import importlib.util,sys;"
             "sys.exit(0 if importlib.util.find_spec('ttnn') else 1)"],
            timeout=30,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def legacy_serve_available() -> bool:
    """Whether the legacy-runner server (``tt_kernel.legacy_serve``) can actually run here.

    The module itself always imports (it ships with tt-model), so what matters is its
    web-server dependencies. DETECTION only — ``find_spec`` never imports them.
    """
    try:
        return (importlib.util.find_spec("fastapi") is not None
                and importlib.util.find_spec("uvicorn") is not None)
    except (ImportError, ValueError):
        return False


def runner_spec_importable(spec: str, python: Optional[str] = None) -> bool:
    """Whether a *reference* runner's module is importable in the target interpreter.

    Used by ``pull`` to verify a not-shipped (reference) runner is actually present
    before claiming the bundle is ready. The module is the part of ``spec`` before the
    ``:`` (``"pkg.mod:Runner"``) or the dotted prefix (``"pkg.mod.Runner"``). DETECTION
    only — ``find_spec`` never imports the module. Mirrors ``ttnn_importable``: checks
    this process directly for the default interpreter, else shells out.
    """
    module = spec.split(":", 1)[0] if ":" in spec else spec.rsplit(".", 1)[0]
    if python is None or python == sys.executable:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False
    try:
        proc = subprocess.run(
            [python, "-c", "import importlib.util,sys;"
             f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"],
            timeout=30,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def serve_argv(
    model: str,
    *,
    runner_spec: str,
    python: Optional[str] = None,
) -> List[str]:
    """Argv for the legacy-runner server — ``tt_kernel.legacy_serve``.

    ``runner_spec`` is required: the shim serves one specific runner (there is no dynamic
    / bare-repo path anymore — that was the retired dispatch runtime). ``model`` is the
    local weights dir the runner loads.
    """
    return [python or "python", "-m", _LEGACY_SERVE_MODULE,
            "--runner", runner_spec, "--model", str(model)]


def serve_command(runner_spec: str, weights_path: Path) -> str:
    """The exact ready-to-run line for the legacy-runner OpenAI server."""
    return " ".join(serve_argv(str(weights_path), runner_spec=runner_spec))


# ----------------------------------------------------------------- self-contained
def install_self_contained(bundle_dir: Path, venv_dir: Path) -> Path:
    """Run a v5 self-contained bundle's ``install.sh`` to build its own venv.

    The generated ``install.sh`` creates ``venv_dir``, pip-installs the shipped wheels
    (ttnn with the author's kernels + base vLLM + plugin), then the deps from
    ``requirements.txt``. Returns the venv's python. Raises CalledProcessError on failure.
    This is the "install the platform" step that makes a package need only a card + firmware.
    """
    script = bundle_dir / "install.sh"
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found (not a self-contained bundle).")
    subprocess.run(["bash", str(script), str(venv_dir)], check=True)
    return venv_dir / "bin" / "python"


# --------------------------------------------------------------------------- vLLM
def vllm_available(python: Optional[str] = None) -> bool:
    """Whether the Tenstorrent vLLM serving stack (vLLM + the plugin) is importable.

    DETECTION only (``find_spec``) — never imports vLLM. Both the fork and the plugin must be
    present for the serve handoff to work. With ``python`` set (a selected tt-metal instance's
    interpreter, which may be a different venv than the one running tt-model), the check
    shells out to that interpreter — otherwise a pipx-installed tt-model would report the
    stack missing even though the *pinned* build can serve.
    """
    if python is not None and python != sys.executable:
        try:
            proc = subprocess.run(
                [python, "-c", "import importlib.util as u,sys;"
                 f"sys.exit(0 if u.find_spec('{_VLLM_PKG}') and "
                 f"u.find_spec('{_VLLM_PLUGIN_PKG}') else 1)"],
                timeout=30,
            )
            return proc.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False
    try:
        return (
            importlib.util.find_spec(_VLLM_PKG) is not None
            and importlib.util.find_spec(_VLLM_PLUGIN_PKG) is not None
        )
    except (ImportError, ValueError):
        return False


def vllm_serve_env(bundles_dir: Path, launch_env: Optional[dict] = None,
                   *, activation_env: Optional[dict] = None) -> dict:
    """The full environment for a vLLM serve subprocess.

    Layering (later wins): the current process env, then the selected tt-metal instance's
    ``activation_env`` (``TT_METAL_HOME`` / ``PYTHONPATH`` / ``LD_LIBRARY_PATH`` — so a stale
    ambient ``TT_METAL_HOME`` is overridden to point at the pinned build), then
    ``EXTRA_MODELS_DIR`` (so the plugin discovers the pulled model), then the bundle's
    per-machine launch env (``MESH_DEVICE``, ``VLLM_USE_V1``, … — always authoritative last).
    """
    env = dict(os.environ)
    if activation_env:
        for k, v in activation_env.items():
            k, v = str(k), str(v)
            # Path-list vars from a pinned build must ADD to, not replace, the inherited value
            # (dropping the system LD_LIBRARY_PATH can break the loader); everything else
            # (TT_METAL_HOME, MESH_DEVICE, …) is an authoritative override.
            if k in _PREPEND_ENV and env.get(k) and v not in env[k].split(os.pathsep):
                env[k] = v + os.pathsep + env[k]
            else:
                env[k] = v
    env[ENV_EXTRA_MODELS_DIR] = str(bundles_dir)
    if launch_env:
        env.update({str(k): str(v) for k, v in launch_env.items()})
    return env


# Env vars that are colon-separated path lists — a pinned instance prepends to them.
_PREPEND_ENV = ("PYTHONPATH", "LD_LIBRARY_PATH")


def is_python_command(token: str) -> bool:
    """True if ``token`` is a Python interpreter invocation the pin can safely replace.

    Matches ``python``, ``python3``, ``python3.10``, and absolute/relative paths whose
    basename is one of those (``/opt/tt/build/python_env/bin/python``). Does NOT match
    ``vllm``, ``bash``, etc. — replacing those with an interpreter would be wrong, so the
    caller warns instead of silently mis-pinning.
    """
    return bool(re.match(r"^python[0-9.]*$", os.path.basename(str(token))))


def vllm_serve_argv(launch_command: List[str], *, python: Optional[str] = None) -> List[str]:
    """The argv to launch the vLLM OpenAI server, from a bundle's per-machine command.

    The bundle's ``launch.command`` is authoritative (e.g. ``["python3",
    "server_example_tt.py", "--model", ...]``). When ``python`` (a pinned instance's
    interpreter) is given, it replaces the command's first token whenever that token *is* a
    Python interpreter — bare, versioned, or an absolute path (see ``is_python_command``) — so
    the pin isn't silently dropped for the common ``python3.10`` / abs-path forms. A non-Python
    launcher (``vllm``, ``bash``) can't be substituted; the caller detects that and warns.
    """
    argv = [str(c) for c in launch_command]
    if python and argv and is_python_command(argv[0]):
        argv[0] = python
    return argv


def health_check(base_url: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    """Probe an OpenAI-compatible server's ``/v1/models`` (cheap liveness check).

    Returns ``(ok, detail)``. Uses only the stdlib so tt-model adds no HTTP dependency.
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — localhost health probe
            code = resp.getcode()
            return (200 <= code < 300, f"GET {url} -> {code}")
    except urllib.error.URLError as exc:
        return (False, f"GET {url} failed: {exc}")
    except (OSError, ValueError) as exc:
        return (False, f"GET {url} failed: {exc}")


__all__ = [
    "resolve_models_dir",
    "download_weights",
    "pip_install_wheels",
    "ttnn_importable",
    "runner_spec_importable",
    "legacy_serve_available",
    "serve_argv",
    "serve_command",
    "vllm_available",
    "vllm_serve_env",
    "vllm_serve_argv",
    "health_check",
    "ENV_EXTRA_MODELS_DIR",
]


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Whether something is already listening on ``port``.

    One syscall, run before launch. Without it, `serve` spent ~18 seconds loading the vLLM
    plugin and ttnn and then handed the user a 20-frame traceback from inside vLLM
    (``OSError: [Errno 98] Address already in use``) for a condition knowable up front.

    Deliberately only *detects*. tt-model does not own whatever holds the port, so it names
    the problem and stops — it does not offer to kill a process it knows nothing about.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return True
    return False


def port_of(endpoint: str) -> Optional[int]:
    """The port from a ``http://host:port`` endpoint string, or None."""
    try:
        return int(endpoint.rsplit(":", 1)[1].split("/")[0])
    except (IndexError, ValueError):
        return None


def module_importable(module: str, python: Optional[str] = None,
                      search_paths: Optional[List[str]] = None) -> bool:
    """Whether ``module`` can be found by the interpreter that will run the server.

    ``find_spec``, not ``import``: the target modules pull in the whole device stack, and
    the question here is only whether the code is present.

    ``search_paths`` must include anything the runtime will add to ``sys.path`` — for us,
    the bundle folder. A bundle may ship its own ``models/`` subtree (the TT plugin resolves
    adapters relative to each ``EXTRA_MODELS_DIR`` entry), so checking the bare interpreter
    would call a perfectly servable bundle broken.
    """
    import subprocess as sp
    import sys as _sys

    prelude = "".join(f"sys.path.insert(0, {p!r})\n" for p in (search_paths or []))
    code = (
        "import sys\n"
        + prelude
        + "import importlib.util as u\n"
        f"sys.exit(0 if u.find_spec({module!r}) else 1)\n"
    )
    try:
        return sp.run([python or _sys.executable, "-c", code],
                      capture_output=True, timeout=120).returncode == 0
    except (OSError, sp.SubprocessError):
        return True  # cannot tell; do not invent a blocker


def adapter_module(main_class: Optional[str]) -> Optional[str]:
    """The full module path of a bundle's ``main_class``.

    `vllm_metadata.json` names the serving adapter as
    ``models.autoports.qwen_qwen3_32b.tt.generator_vllm:Qwen3ForCausalLM``.
    """
    if not main_class:
        return None
    return (main_class.split(":", 1)[0].strip() or None)


def adapter_root(main_class: Optional[str]) -> Optional[str]:
    """The top-level package of a bundle's ``main_class``."""
    module = adapter_module(main_class)
    return module.split(".", 1)[0] if module else None


def missing_adapter_segment(main_class: Optional[str], python: Optional[str] = None,
                            search_paths: Optional[List[str]] = None) -> Optional[str]:
    """The first dotted segment of the adapter that will not import, or None if it all does.

    Checking only the top-level package was too coarse and produced a wrong answer on a real
    box: putting a tt-metal checkout on PYTHONPATH makes ``models`` importable, so a
    root-only check goes green — while ``models.autoports`` still does not exist there and
    vLLM still dies on the import. Reporting the deepest prefix that *does* resolve is also
    what makes the message actionable: "models resolves but models.autoports does not" says
    the tree is present and this adapter is not, which is a different fix from "no models
    tree at all".
    """
    module = adapter_module(main_class)
    if not module:
        return None
    parts = module.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if not module_importable(prefix, python, search_paths):
            return prefix
    return None
