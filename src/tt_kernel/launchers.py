# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""How a container package is launched — one class per ``kind``.

A *kind* is the serving stack inside the image and the command that starts it. It is the
only stack-specific knowledge in the container path: the Dockerfile is kind-agnostic and
everything that varies arrives through the hooks below, so adding a kind is a new class
here plus a registration in :data:`KINDS` — no change to the manifest schema, the image
build, or any command.

**There is one kind today: ``vllm``, the ``tenstorrent/vllm`` fork with its in-tree
plugin.** That is deliberately the same thing this repo has always meant by ``vllm``:
``provision.py`` clones ``tenstorrent/vllm@dev``, ``toolchain.py`` requires
"tenstorrent/vllm@dev + plugin", and a v4 manifest's ``runtime.kind = "vllm"`` describes
the plugin as the package "the fork ships alongside it". Reusing the word for anything
else would give one field two meanings.

A second stack — stock ``vllm==X.Y.Z`` from PyPI plus the standalone public
``tenstorrent/vllm-tt-plugin`` — is a real and different way to serve, and it is what a
model built against a released vLLM would want. It is NOT supported here yet: nothing in
``install``/``doctor``/v4/v5 provisions it, so shipping it would mean the container path
alone understood a stack the rest of the tool could not. When a model needs it, it is a
new class in this module (``vllm-stock``) plus a row in :data:`KINDS`.

Two different inputs on purpose:

* ``validate`` runs at AUTHORING time against the YAML the author wrote, because that is
  where a bad ``runtime:`` block should be caught — before a multi-hour build, not after.
* ``serve_argv`` / ``serve_env`` / ``ready_probe`` run at CONSUME time against the
  published manifest, which is all a consumer has.

Argv composition is pure and total: no environment, no filesystem, no clock. That is what
lets ``serve --print`` be the test surface for every flag we claim to pass.

``install_lines`` and ``verify_lines`` are the BUILD side: they render the two generated
scripts (``install_engine.sh``, ``verify.sh``) that the Dockerfile runs, which is why the
Dockerfile itself never mentions vLLM and never changes per kind.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .manifest import Manifest, ServeProfile

if TYPE_CHECKING:
    from .container_manifest import ContainerManifest


class LauncherError(ValueError):
    """A launch configuration that must not proceed. The message is user-facing."""


def _weights_id(m: Manifest) -> str:
    if m.weights is None:
        raise LauncherError("a container package must reference weights by HF repo id")
    return m.weights.repo_id


class VllmLauncher:
    """``kind: vllm`` — the ``tenstorrent/vllm`` fork with its in-tree plugin.

    Launched through tt-metal's readiness runner rather than ``vllm serve``: models on
    this stack expect that runner's flag names (``--tt-config``,
    ``--additional-server-args``, ``--server-timeout``), and it resolves the plugin's
    config flag against the installed vLLM, forwards an explicit mesh grid, and hands off
    to a stock ``vllm.entrypoints.openai.api_server``.

    Both the fork and its in-tree plugin install *editable* (the fork's own documented
    install), so the fork checkout must survive into the runtime image — about 200 MB.
    """

    name = "vllm"

    #: keys the manifest's ``runtime:`` block may contain for this kind
    RUNTIME_KEYS = ("vllm", "extension", "lock", "model_dir")

    #: the log line whose appearance means the OpenAI server is accepting requests
    READY_LINE = "Server ready at"

    #: where the fork checkout lives in both build stages
    FORK_DIR = "/opt/vllm"

    #: vLLM's PyPI metadata is generated on a CUDA machine; without the CPU index a plain
    #: install resolves the CUDA dependency set (~4 GB of nvidia-* wheels, no device here)
    PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

    def validate(self, m: "ContainerManifest") -> None:
        from .container_manifest import ContainerManifestError

        rt = m.runtime
        vllm = rt.get("vllm") or {}

        if vllm.get("version") and not vllm.get("repo"):
            # The stock-vLLM shape. Say so precisely instead of "missing repo/ref": an
            # author who wrote this was describing a real stack, just not one we serve.
            raise ContainerManifestError(
                "runtime.vllm.version describes stock vLLM from PyPI, which tt-model does "
                "not serve yet — the only supported stack is the tenstorrent/vllm fork. "
                "Give runtime.vllm.{repo, ref} instead, pinning the sha the model was "
                "VALIDATED with."
            )
        if not (vllm.get("repo") and vllm.get("ref")):
            raise ContainerManifestError(
                "kind vllm requires runtime.vllm.{repo, ref} — the tenstorrent/vllm fork, "
                "pinned to the sha the model was VALIDATED with (the plugin is in-tree)"
            )
        if rt.get("plugin"):
            raise ContainerManifestError(
                "kind vllm takes no runtime.plugin: the plugin comes from the fork's own "
                "plugins/vllm-tt-plugin"
            )

        model_dir = rt.get("model_dir")
        if not model_dir:
            raise ContainerManifestError(
                "kind vllm requires runtime.model_dir (the model's directory in the "
                "tt-metal tree, e.g. models/autoports/<name>) — the readiness runner "
                "launches by --model-dir"
            )
        # The launcher resolves model_dir INSIDE the image, so it must be covered by the
        # allowlist that decides what gets into the image at all.
        covered = any(
            model_dir == c or model_dir.startswith(c.rstrip("/") + "/") for c in m.source.code
        )
        if not covered:
            raise ContainerManifestError(
                f"runtime.model_dir {model_dir!r} is not covered by source.code — the "
                "launcher would not find the model inside the image"
            )

        for key in rt:
            if key not in self.RUNTIME_KEYS:
                raise ContainerManifestError(
                    f"kind vllm does not understand runtime.{key}; expected one of "
                    + ", ".join(self.RUNTIME_KEYS)
                )

    # ---- image build ---------------------------------------------------------------

    def install_lines(self, m: "ContainerManifest") -> List[str]:
        """Shell lines that install the serving stack into the image's venv, after
        tt-metal itself is built and installed."""
        rt = m.runtime
        vllm = rt["vllm"]
        ref = vllm.get("sha") or vllm["ref"]
        lines: List[str] = [
            f"git clone {shlex.quote(vllm['repo'])} {self.FORK_DIR}"
            f" && git -C {self.FORK_DIR} checkout {shlex.quote(ref)}",
        ]
        if rt.get("lock"):
            # The lock IS the dependency set, so nothing resolves at build time and two
            # builds a week apart produce the same environment.
            lines.append(
                'uv pip install --python "$VENV/bin/python" -r /ctx/requirements.lock '
                f"--extra-index-url {self.PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match"
            )
            deps = "--no-deps "
        else:
            deps = ""
        lines += [
            # Editable, matching the fork's own documented install; the checkout is COPY'd
            # into the runtime stage so the .pth files stay valid.
            f'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" {deps}'
            f"-e {self.FORK_DIR} --extra-index-url {self.PYTORCH_CPU_INDEX} "
            f"--index-strategy unsafe-best-match",
            f'uv pip install --python "$VENV/bin/python" -e {self.FORK_DIR}/plugins/vllm-tt-plugin',
            # The editable installs bake build-stage paths; drop VCS metadata only.
            f"rm -rf {self.FORK_DIR}/.git",
        ]
        if rt.get("extension"):
            # The model's own vLLM extension ships inside code/, so install it from the
            # staged tree already present in the image.
            lines.append(
                f'uv pip install --python "$VENV/bin/python" /opt/tt-metal/{rt["extension"]}'
            )
        return lines

    def verify_lines(self, m: "ContainerManifest") -> List[str]:
        """Assertions run INSIDE the finished image. These are what make the prune and
        the code/ allowlist safe: an under-shipped image fails here, on the author's
        machine, not on a consumer's first boot."""
        checks = [
            "import ttnn, vllm, vllm_tt_plugin",
            "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__",
            f"import vllm; assert vllm.__file__.startswith('{self.FORK_DIR}'), vllm.__file__",
            "import models.common.readiness_check.run_vllm_server",
        ]
        # torch is a TRANSITIVE dependency of vLLM here (tt-metal installs none), so
        # nothing makes it agree with what ttnn was built against unless we check.
        pin = metal_torch_pin(_local_metal_tree(m))
        if pin:
            checks.append(
                f"import torch; v = torch.__version__.split('+')[0]; "
                f"assert v == {pin!r}, "
                f"f'torch {{v}} was resolved by vLLM but tt-metal pins {pin}; "
                f"ttnn extension modules were built against {pin} — pin it via "
                f"runtime.lock, or align runtime.vllm.ref'"
            )
        if m.runtime.get("extension"):
            checks.append(
                "import os; md = os.environ['EXTRA_MODELS_DIR']; "
                "entries = [e for e in os.listdir(md) "
                "if os.path.exists(os.path.join(md, e, 'vllm_metadata.json'))]; "
                "assert entries, f'EXTRA_MODELS_DIR {md} registers no models'"
            )
        lines = [f'"$VENV/bin/python" -c {shlex.quote("; ".join(checks))}']
        # the model author's own assertions, from the manifest's verify: list
        lines += [f'"$VENV/bin/python" -c {shlex.quote(v)}' for v in m.verify]
        return lines

    # ---- serve -----------------------------------------------------------------------

    def serve_argv(self, m: Manifest, profile: ServeProfile) -> List[str]:
        from .container_manifest import parse_mesh_device

        rows, cols = parse_mesh_device(profile.mesh_device or "")
        model_dir = (m.container.runtime if m.container else {}).get("model_dir")
        if not model_dir:
            raise LauncherError("kind vllm requires runtime.model_dir")
        argv = [
            "python", "-m", "models.common.readiness_check.run_vllm_server",
            "--stages", "serve",
            "--model-dir", str(model_dir),
            "--hf-model", _weights_id(m),
            "--mesh-device", f"({rows}, {cols})",
            "--max-num-seqs", str(profile.max_num_seqs),
        ]
        if profile.max_model_len is not None:
            argv += ["--max-model-len", str(profile.max_model_len)]
        argv += ["--block-size", str(profile.block_size)]
        if profile.server_timeout is not None:
            argv += ["--server-timeout", str(profile.server_timeout)]
        argv += ["--port", str(profile.port or 8000)]
        tt_cfg = profile.additional_config.get("tt")
        if tt_cfg:
            argv += ["--tt-config", json.dumps(tt_cfg)]
        # This runner takes extra server flags as ONE joined string, not as loose argv.
        extra = profile.flat_args() + _capability_argv(profile)
        if extra:
            argv += ["--additional-server-args", " ".join(extra)]
        return argv

    def serve_env(self, m: Manifest, profile: ServeProfile) -> Dict[str, str]:
        # The mesh goes through --mesh-device here, NOT through a MESH_DEVICE env var.
        # tt_transformers-style adapters read the model id from HF_MODEL, not from
        # vLLM's --model.
        env = {"HF_MODEL": _weights_id(m)}
        env.update(profile.env)
        return env

    def ready_probe(self, m: Manifest) -> str:
        return self.READY_LINE


# tt-metal declares the torch it expects in its dev requirements, e.g.
#   --extra-index-url https://download.pytorch.org/whl/cpu
#   torch==2.11.0 ; platform_machine == 'x86_64'
_TORCH_PIN_RE = re.compile(
    r"^torch==(?P<version>[^\s;#]+)\s*(;.*platform_machine\s*==\s*'x86_64')?\s*$"
)

METAL_TORCH_REQUIREMENTS = "tt_metal/python_env/requirements-dev.txt"


def _local_metal_tree(m: "ContainerManifest") -> Optional[Path]:
    """The author's tt-metal checkout, or None when the manifest names a git source."""
    src = m.source.tt_metal
    return Path(src) if isinstance(src, str) else None


def metal_torch_pin(metal_tree: Optional[Path]) -> Optional[str]:
    """The torch version tt-metal declares it wants, or None if it cannot be read.

    tt-metal's ``setup.py`` has no ``install_requires``, so installing it brings NO
    torch: torch arrives only as a transitive dependency of vLLM. Nothing therefore
    forces the two to agree, and a vLLM release that resolves a different torch than
    ttnn's extension modules were built against breaks at import or, worse, at device
    open. Reading the pin lets the image assert the agreement at BUILD time.

    Returns None for a git-mode source (no local tree yet) or an unreadable/changed
    requirements file — the check is then skipped rather than guessed at.
    """
    if metal_tree is None:
        return None
    req = Path(metal_tree) / METAL_TORCH_REQUIREMENTS
    try:
        text = req.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        m = _TORCH_PIN_RE.match(line.strip())
        if m:
            return m.group("version")
    return None


def _capability_argv(profile: ServeProfile) -> List[str]:
    """Render the tool/reasoning parsers, matching the v4 path's rules exactly.

    ``--tool-call-parser`` is emitted WITH ``--enable-auto-tool-choice`` because vLLM
    hard-errors on the former without the latter, and ``--reasoning_parser`` keeps its
    underscore: typer normalises '_'->'-' so '--tool_parser' would become the nonexistent
    '--tool-parser', while '--reasoning_parser' normalises to the valid spelling.
    See ``bundles._compose_launch_vllm``.
    """
    cap = profile.capabilities
    if cap is None:
        return []
    argv: List[str] = []
    if cap.tool_parser:
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", cap.tool_parser]
    if cap.reasoning_parser:
        argv += ["--reasoning_parser", cap.reasoning_parser]
    return argv


KINDS: Dict[str, object] = {VllmLauncher.name: VllmLauncher()}


def launcher_for(kind: str):
    try:
        return KINDS[kind]
    except KeyError:
        raise LauncherError(
            f"unsupported kind {kind!r}; supported: " + ", ".join(sorted(KINDS))
        ) from None
